from copy import deepcopy

import numpy
import random
import datetime
import tensorflow

from typing import List
from api.regras.DatasetRegras import DatasetRegras

from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import MSE, MeanSquaredError, KLDivergence
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, RMSprop
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, SimpleRNN, Dropout


class ParamsRede:
    def __init__(self):
        self.isDadosUmLadoSo: bool = False
        self.idTypeReturn: int = 1
        self.idTypeRotulo: int = 1
        self.idTypeEntrada: int = 4
        self.tpLimitHistoricoMedias: tuple = (3, 7)
        self.tpQtdeDadosEntradaForTeam: tuple = (16, 26)
        self.qtdeTimesForTime: int = 1
        self.qtdeDadosValidarDataset: int = 0
        self.qtdeDadosValidarTreino: int = 5
        self.isRecurrent: bool = True
        self.nTentativas: int = 10
        self.nRedes: int = 1
        self.tpNCadamasForRede: tuple = (1, 1)
        self.isLSTM: bool = False


class KerasNeuralInteligente:
    def __init__(self):
        self.ignoreIntelisense = False

    def obterDataset(self, paramsRede: ParamsRede, arrIdsTeam: list, arrIdsExpecficos: list):
        if len(arrIdsTeam) == 0:
            raise Exception("e preciso passar um id team pelo menos")

        datasetRegra = DatasetRegras()
        limitHistoricoMedias = random.randint(paramsRede.tpLimitHistoricoMedias[0],
                                              paramsRede.tpLimitHistoricoMedias[1])
        qtdeDadosEntradaForTeam = random.randint(paramsRede.tpQtdeDadosEntradaForTeam[0],
                                                 paramsRede.tpQtdeDadosEntradaForTeam[1])

        datasetEntrada, datasetRotulo, datasetPrever = datasetRegra.obterDataset(
            arrIdsTeam=arrIdsTeam, limitHistorico=limitHistoricoMedias,
            qtdeDadosForTeam=qtdeDadosEntradaForTeam, arrIdsExpecficos=arrIdsExpecficos,
            idTypeReturn=paramsRede.idTypeReturn, isDadosUmLadoSo=paramsRede.isDadosUmLadoSo,
            idTypeRotulo=paramsRede.idTypeRotulo, idTypeEntrada=paramsRede.idTypeEntrada)

        for idxEntrada in range(len(datasetEntrada)):
            if ((len(datasetEntrada[idxEntrada]) <
                 (qtdeDadosEntradaForTeam * len(arrIdsExpecficos)) - len(datasetPrever[idxEntrada]))):
                raise Exception("Sem dado suficientes temos somente: " + str(len(datasetEntrada[idxEntrada])))
        else:
            print("Vai ser somente com " + str(len(datasetEntrada)) + " com idType: ", paramsRede.idTypeReturn)

        entradasRede, rotulosRede, preverRede = self.obterDatasetForRecurrent(
            paramsRede=paramsRede, datasetEntrada=datasetEntrada, datasetRotulo=datasetRotulo,
            datasetPrever=datasetPrever)

        return entradasRede, rotulosRede, preverRede

    def obterDatasetForRecurrent(self, paramsRede: ParamsRede, datasetEntrada: list, datasetRotulo: list,
                                 datasetPrever: list):
        self.ignoreIntelisense = True
        arrEntradas = [[] for _ in datasetEntrada]
        arrPrever = [[] for _ in datasetRotulo]
        arrRotulos = [[] for _ in datasetPrever]

        for idxType in range(len(datasetEntrada)):
            for idx in range(len(datasetEntrada[idxType])):
                if paramsRede.isRecurrent:
                    arrEntradas[idxType].append([datasetEntrada[idxType][idx]])
                else:
                    arrEntradas[idxType].append(datasetEntrada[idxType][idx])

                # arrRotulos.append(rotulos[i][0]) para a antiga dataset
                arrRotulos[idxType].append(datasetRotulo[idxType][idx])

            for idx in range(len(datasetPrever[idxType])):
                if paramsRede.isRecurrent:
                    arrPrever[idxType].append([datasetPrever[idxType][idx]])
                else:
                    arrPrever[idxType].append(datasetPrever[idxType][idx])

        return arrEntradas, arrRotulos, arrPrever

    def treinarRede(self, arrIdsTeam: list, arrIdsExpecficos: List[int]):
        paramsRede = ParamsRede()
        arrPredicts, arrAcertos = self.obterMelhorTentativa(paramsRede=paramsRede, arrIdsTeam=arrIdsTeam,
                                                            arrIdsExpecficos=arrIdsExpecficos)
        return arrPredicts, arrAcertos

    def obterMelhorTentativa(self, paramsRede: ParamsRede, arrIdsTeam: list, arrIdsExpecficos: List[int]):

        arrBestRede = []
        arrBestAcertos = []
        arrIdxEntradaParar = []
        isPararTentativa = False
        arrPrevisoesTypeEntradas = []

        for idxTentativa in range(paramsRede.nTentativas):
            if isPararTentativa:
                break
            dateTentativaInicio = datetime.datetime.now()
            arrEntradas, arrRotulos, arrPrevisoes = self.obterDataset(paramsRede=paramsRede, arrIdsTeam=arrIdsTeam,
                                                                      arrIdsExpecficos=arrIdsExpecficos)

            if idxTentativa == 0:
                arrBestRede = [Sequential() for _ in arrEntradas]
                arrBestAcertos = [[] for _ in arrEntradas]
                arrPrevisoesTypeEntradas = [[] for _ in arrEntradas]

            for idxTypeEntrada in range(len(arrEntradas)):
                isPrever = False
                if idxTypeEntrada in arrIdxEntradaParar:
                    continue

                arrEntrada = arrEntradas[idxTypeEntrada]
                arrRotulo = arrRotulos[idxTypeEntrada]
                arrPrevisao = arrPrevisoes[idxTypeEntrada]

                bestRede, qtdeAcertos, allPredicts = self.obterMelhorRede(
                    paramsRede=paramsRede, datasetEntrada=arrEntrada, datasetRotulo=arrRotulo)

                if idxTentativa == 0:
                    arrBestRede[idxTypeEntrada] = bestRede
                    arrBestAcertos[idxTypeEntrada] = qtdeAcertos
                    isPrever = True
                else:
                    perctAcertoAtual = qtdeAcertos[0] / sum(qtdeAcertos)
                    perctBestAcertoAtual = arrBestAcertos[idxTypeEntrada][0] / sum(arrBestAcertos[idxTypeEntrada])
                    if ((perctAcertoAtual > perctBestAcertoAtual) or
                            (perctBestAcertoAtual <= 0.6 and perctAcertoAtual <= 0.2)):
                        arrBestRede[idxTypeEntrada] = bestRede
                        arrBestAcertos[idxTypeEntrada] = qtdeAcertos
                        isPrever = True

                if qtdeAcertos[0] / sum(qtdeAcertos) >= 1 and idxTypeEntrada not in arrIdxEntradaParar:
                    arrIdxEntradaParar.append(idxTypeEntrada)

                if len(arrIdxEntradaParar) == len(arrEntradas):
                    isPararTentativa = True

                if isPrever:
                    previsao = arrBestRede[idxTypeEntrada].predict(x=numpy.array(arrPrevisao)).tolist()
                    arrPrevisoesTypeEntradas[idxTypeEntrada] = previsao

            dateTentativaFim = datetime.datetime.now()
            diferenceSeconds = (dateTentativaFim - dateTentativaInicio).total_seconds()
            if diferenceSeconds < 60:
                msgTempo = "levou {} segundos".format(diferenceSeconds)
            else:
                diferenceMinutos = diferenceSeconds / 60
                msgTempo = "levou {} minutos".format(diferenceMinutos)
            self.gravarLogs(msg="tentativa {} / {} {} / {}\n".format(idxTentativa, paramsRede.nTentativas,
                                                                     msgTempo, arrBestAcertos))

        return arrPrevisoesTypeEntradas, arrBestAcertos

    def obterMelhorRede(self, paramsRede: ParamsRede, datasetEntrada: list, datasetRotulo: list):
        nNeuronsEntrada = len(datasetEntrada[0][0]) if paramsRede.isRecurrent else len(datasetEntrada[0])
        nNeuronsRotulo = len(datasetRotulo[0])
        nItensForTime = len(datasetEntrada[0]) if paramsRede.isRecurrent else None
        nNeuronsForCamada = nNeuronsEntrada + nNeuronsRotulo
        arrTpNNeuronsForCamada = [(int(nNeuronsForCamada), nNeuronsForCamada) for _ in range(paramsRede.tpNCadamasForRede[1])]
        funcAtivacao = "sigmoid" if len(datasetRotulo[0]) == 1 else "softmax"

        arrRedes = self.buildRedesNeurais(paramsRede=paramsRede, nNeuronsEntrada=nNeuronsEntrada,
                                          nNeuronsRotulo=nNeuronsRotulo, funcAtivacao=funcAtivacao,
                                          arrTpNNeuronsForCamada=arrTpNNeuronsForCamada,
                                          nItensForTime=nItensForTime)

        if funcAtivacao == "softmax":
            functionLoss = "categorical_crossentropy"
        elif funcAtivacao == "sigmoid":
            functionLoss = "binary_crossentropy"
        else:
            functionLoss = "mse"

        bestRede: Sequential = None
        bestQtdeAcerto: list = None

        for idxRede in range(len(arrRedes)):
            redeNeural = arrRedes[idxRede]
            bkpDatasetEntrada = deepcopy(datasetEntrada)
            bkpDatasetRotulo = deepcopy(datasetRotulo)
            batchSize = len(bkpDatasetEntrada)

            arrValidarEntrada = bkpDatasetEntrada[-paramsRede.qtdeDadosValidarTreino:]
            arrValidarRotulo = bkpDatasetRotulo[-paramsRede.qtdeDadosValidarTreino:]

            arrTreinarEntrada = bkpDatasetEntrada[:-paramsRede.qtdeDadosValidarTreino]
            arrTreinarRotulo = bkpDatasetRotulo[:-paramsRede.qtdeDadosValidarTreino]

            # funcStoppingA = EarlyStopping(monitor="accuracy", patience=500, mode="max")
            funcStoppingA = StopTrainingAtAccuracy(target_accuracy=1.0)
            redeNeural.compile(optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"], loss=functionLoss)
            redeNeural.fit(x=numpy.array(arrTreinarEntrada), y=numpy.array(arrTreinarRotulo),
                           callbacks=[funcStoppingA], epochs=2000, batch_size=batchSize)

            arrQtdeAcertos = [0, 0]
            isAcertouUltimoRotulo = False
            for idxValidarentrada in range(len(arrValidarEntrada)):
                validarEntrada = [arrValidarEntrada[idxValidarentrada]]
                validarRotulo = arrValidarRotulo[idxValidarentrada]

                arrPredictRede = redeNeural.predict(x=numpy.array(validarEntrada))

                for idxPredicRede in range(len(arrPredictRede)):
                    predict = arrPredictRede[idxPredicRede]
                    if funcAtivacao == "softmax":
                        maxIndexRotulo = numpy.argmax(validarRotulo)
                        maxIndexPredict = numpy.argmax(predict)

                        if maxIndexPredict == maxIndexRotulo:
                            arrQtdeAcertos[0] += 1
                            isAcertouUltimoRotulo = True
                        else:
                            arrQtdeAcertos[1] += 1
                            isAcertouUltimoRotulo = False
                    else:
                        throbleshot = 0.5
                        predictNorm = int(predict[0] >= throbleshot)
                        if predictNorm == validarRotulo[0]:
                            arrQtdeAcertos[0] += 1
                            isAcertouUltimoRotulo = True
                        else:
                            arrQtdeAcertos[1] += 1
                            isAcertouUltimoRotulo = False

                for idxValEntrada in range(len(validarEntrada)):
                    arrTreinarEntrada.append(validarEntrada[idxValEntrada])
                    arrTreinarRotulo.append(validarRotulo)

                # funcStoppingB = EarlyStopping(monitor="accuracy", patience=300, mode="max")
                funcStoppingB = StopTrainingAtAccuracy(target_accuracy=1.0)
                redeNeural.fit(x=numpy.array(arrTreinarEntrada), y=numpy.array(arrTreinarRotulo),
                               callbacks=[funcStoppingB], epochs=500, batch_size=batchSize)

            if bestRede is None:
                bestRede = arrRedes[idxRede]
                bestQtdeAcerto = arrQtdeAcertos
            else:
                perctAcertoAtual = arrQtdeAcertos[0] / sum(arrQtdeAcertos)
                perctBestAcertoAtual = bestQtdeAcerto[0] / sum(bestQtdeAcerto)
                if ((perctAcertoAtual > perctBestAcertoAtual) or
                        (perctAcertoAtual >= perctBestAcertoAtual and isAcertouUltimoRotulo) or
                        (perctBestAcertoAtual <= 0.6 and perctAcertoAtual <= 0.2)):
                    bestRede = arrRedes[idxRede]
                    bestQtdeAcerto = arrQtdeAcertos

            if arrQtdeAcertos[0] / sum(arrQtdeAcertos) >= 1:
                break

        allPredict: list = numpy.array(bestRede.predict(x=numpy.array(datasetEntrada))).tolist()
        return bestRede, bestQtdeAcerto, allPredict

    @staticmethod
    def buildRedesNeurais(paramsRede: ParamsRede, nNeuronsEntrada: int, nNeuronsRotulo: int,
                          funcAtivacao: str, arrTpNNeuronsForCamada: List[tuple], nItensForTime: int):
        """
            nItensForTime deve ser o numero de times se a rede for recurrent senão deve ser None
            arrTpNNeuronsForCamada deve ser o maximo de itens no tpNCadamasForRede
        """
        tensorflow.compat.v1.keras.backend.clear_session()
        arrRedes = []

        for idxRede in range(paramsRede.nRedes):
            modelSequencial = Sequential()
            nCamadas = random.randint(paramsRede.tpNCadamasForRede[0], paramsRede.tpNCadamasForRede[1])

            for idxCamada in range(nCamadas):
                returnSequences = idxCamada < nCamadas - 1
                nNeuronios = random.randint(arrTpNNeuronsForCamada[idxCamada][0],
                                            arrTpNNeuronsForCamada[idxCamada][1])

                if paramsRede.isRecurrent:
                    if idxCamada == 0:
                        inputShape = (nItensForTime, nNeuronsEntrada)
                        if paramsRede.isLSTM:
                            modelSequencial.add(LSTM(units=nNeuronios, activation="tanh",
                                                     input_shape=inputShape, return_sequences=returnSequences,
                                                     kernel_initializer=initializers.GlorotNormal(),
                                                     recurrent_initializer=initializers.GlorotNormal()))
                        else:
                            modelSequencial.add(SimpleRNN(units=nNeuronios, activation="tanh",
                                                          input_shape=inputShape, return_sequences=returnSequences,
                                                          kernel_initializer=initializers.GlorotNormal(),
                                                          recurrent_initializer=initializers.GlorotNormal()))
                    else:
                        if paramsRede.isLSTM:
                            modelSequencial.add(LSTM(units=nNeuronios, activation="tanh",
                                                     return_sequences=returnSequences,
                                                     kernel_initializer=initializers.GlorotNormal(),
                                                     recurrent_initializer=initializers.GlorotNormal(), ))
                        else:

                            modelSequencial.add(SimpleRNN(units=nNeuronios, activation="tanh",
                                                          return_sequences=returnSequences,
                                                          kernel_initializer=initializers.GlorotNormal(),
                                                          recurrent_initializer=initializers.GlorotNormal(),))
                else:
                    inputShape = (nNeuronsEntrada,)
                    if idxCamada == 0:
                        modelSequencial.add(Dense(units=nNeuronios, activation="tanh",
                                                  input_shape=inputShape,
                                                  kernel_initializer=initializers.GlorotNormal()))
                    else:
                        modelSequencial.add(Dense(units=nNeuronios, activation="tanh",
                                                  kernel_initializer=initializers.GlorotNormal()))

            modelSequencial.add(Dense(units=nNeuronsRotulo, activation=funcAtivacao,
                                      kernel_initializer=initializers.GlorotNormal()))
            arrRedes.append(modelSequencial)
        return arrRedes

    @staticmethod
    def gravarLogs(msg: str):
        with open("C:/Users/lucas/OneDrive/Documentos/Projetos/footxap/web/static/js/keras-neural-inteligente.txt",
                  "a", encoding="utf-8") as results:
            results.write(msg)
            results.close()


class StopTrainingAtAccuracy(Callback):
    def __init__(self, target_accuracy=1.0):
        super(StopTrainingAtAccuracy, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')
        if current_accuracy is not None and current_accuracy >= self.target_accuracy:
            print(f"\nAtingiu a precisão desejada de {self.target_accuracy}. Parando o treinamento.")
            self.model.stop_training = True
