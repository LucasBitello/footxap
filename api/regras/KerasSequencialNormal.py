import datetime
from typing import List

import numpy
import random
import tensorflow
from copy import deepcopy

from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, SimpleRNN, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, RMSprop
from tensorflow.keras.losses import MSE, MeanSquaredError, KLDivergence
from tensorflow.keras import initializers

from api.regras.DatasetRegras import DatasetRegras

"""from keras.layers import Dense, LSTM, Input, concatenate, SimpleRNN
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam"""


from api.regras.datasetPartidasRegras import DatasetPartidasRegras
from api.regras.iaUteisRegras import IAUteisRegras


class KerasSequencialNormal:
    def __init__(self):
        self.ignoreIntelisense = True

    def obterDatasetRecurrentForKeras(self, arrIdsTeam: list = [], idTypeReturn: int = 1,
                                      isRecurrent: bool = True, funcAtiv: str = "softmax",
                                      isPassadaTempoDupla: bool = True, qtdeDados: int = 40,
                                      limitHistoricoMedias: int = 5, arrIdsExpecficos: List[int] = [],
                                      isDadosUmLadoSo: bool = True, idTypeRotulo: int = 1, idTypeEntrada: int = 1):
        funcAtivacao = funcAtiv

        if len(arrIdsTeam) == 0:
            raise Exception("e preciso passar um id team pelo menos")

        datasetRegra = DatasetRegras()
        datasetEntrada, datasetRotulo, datasetPrever = datasetRegra.obterDataset(arrIdsTeam=arrIdsTeam,
                                                                                 limitHistorico=limitHistoricoMedias,
                                                                                 qtdeDadosForTeam=qtdeDados,
                                                                                 arrIdsExpecficos=arrIdsExpecficos,
                                                                                 idTypeReturn=idTypeReturn,
                                                                                 isDadosUmLadoSo=isDadosUmLadoSo,
                                                                                 idTypeRotulo=idTypeRotulo,
                                                                                 idTypeEntrada=idTypeEntrada)

        if len(datasetEntrada) < qtdeDados - (len(datasetPrever) + 1):
            raise Exception("Sem dado suficientes temos somente: " + str(len(datasetEntrada)))
        else:
            print("Vai ser somente com " + str(len(datasetEntrada)) + " com idType: ", idTypeReturn)

        entradas, rotulos, prever, datasetValidar = (
            self.obterDatasetEntradaForRecurrent(entrada=datasetEntrada, rotulos=datasetRotulo, prever=datasetPrever,
                                                 qtdeTimesForTime=1, isRecurrent=isRecurrent,
                                                 isPassadaTempoDupla=isPassadaTempoDupla, qtdedadosValidar=0))

        return entradas, rotulos, prever, datasetValidar

    def obterDatasetEntradaForRecurrent(self, entrada, prever, rotulos, qtdeTimesForTime, isRecurrent,
                                        qtdedadosValidar: int = 0, isPassadaTempoDupla: bool = True):
        self.ignoreIntelisense = True
        indexInit = 0
        arrNormalizadoEntrada = []
        arrNormalizadoRotulos = []
        arrNormalizadoPrever = []
        arrTimesEntrada = []
        arrTimesPrever = []
        datasetValidar = None

        if isPassadaTempoDupla:
            idAnterior = 0
            for iEnt in range(len(entrada)):
                arrTimesEntrada.append(entrada[iEnt])
                valValidar = entrada[iEnt][2]
                if (iEnt + 1) % 2 == 0:
                    if valValidar != idAnterior:
                        raise Exception("parece o id jogo anterior e", idAnterior, " e o proximo é ", valValidar)
                    arrNormalizadoEntrada.append(arrTimesEntrada)
                    arrNormalizadoRotulos.append(rotulos[iEnt - 1][0])
                    arrTimesEntrada = []

                else:
                    idAnterior = valValidar

            for iPrev in range(len(prever)):
                arrTimesPrever.append(prever[iPrev])

            if len(arrTimesPrever) != 2:
                raise Exception("Prever com ", len(arrTimesPrever), " e tem q ser 2")

            arrNormalizadoPrever.append(arrTimesPrever)

            datasetValidar = None
            if qtdedadosValidar >= 1:
                datasetValidar = (numpy.array(arrNormalizadoEntrada[-3:]), numpy.array(arrNormalizadoRotulos[-3:]))
                arrNormalizadoEntrada = arrNormalizadoEntrada[:-3]
                arrNormalizadoRotulos = arrNormalizadoRotulos[:-3]

            return arrNormalizadoEntrada, arrNormalizadoRotulos, arrNormalizadoPrever, datasetValidar
        else:
            if qtdeTimesForTime >= 2:

                for indexDado in range(len(entrada) - qtdeTimesForTime):
                    for iEnt in range(len(entrada)):
                        if iEnt < indexInit:
                            pass
                        else:
                            arrTimesEntrada.append(entrada[iEnt])

                            if len(arrTimesEntrada) == qtdeTimesForTime:
                                arrNormalizadoRotulos.append(rotulos[iEnt][0])
                                arrNormalizadoEntrada.append(arrTimesEntrada)
                                indexInit += 1
                                arrTimesEntrada = []
                                break

                arrNormalizadoPrever = []
                arrTimesPrever = arrNormalizadoEntrada[-1]
                arrTimesPrever.pop(0)
                arrTimesPrever.append(prever[0])
                arrNormalizadoPrever.append(arrTimesPrever)

                return arrNormalizadoEntrada, arrNormalizadoRotulos, arrNormalizadoPrever, datasetValidar
            else:
                arrEntradas = []
                arrPrever = []

                arrRotulos = []
                for i in range(len(entrada)):
                    if isRecurrent:
                        arrEntradas.append([entrada[i]])
                    else:
                        arrEntradas.append(entrada[i])

                    # arrRotulos.append(rotulos[i][0]) para a antiga dataset
                    arrRotulos.append(rotulos[i])

                for i in range(len(prever)):
                    if isRecurrent:
                        arrPrever.append([prever[i]])
                    else:
                        arrPrever.append(prever[i])

                if qtdedadosValidar >= 1:
                    datasetValidar = \
                        (numpy.array(arrEntradas[-qtdedadosValidar:]), numpy.array(arrRotulos[-qtdedadosValidar:]))
                    arrNormalizadoEntrada = arrEntradas[:-qtdedadosValidar]
                    arrNormalizadoRotulos = arrRotulos[:-qtdedadosValidar]
                    return arrNormalizadoEntrada, arrNormalizadoRotulos, arrPrever, datasetValidar

                return arrEntradas, arrRotulos, arrPrever, datasetValidar

    def buildRedesNeurais(self, datasetEntrada, datasetRotulo, nRedes: int = 10, funcAtiv: str = "softmax",
                          isRecurrent: bool = True, isDuplaHiposete: bool = False):
        tensorflow.compat.v1.keras.backend.clear_session()
        self.ignoreIntelisense = True
        arrRedesNeurais = []

        for indexRede in range(nRedes):
            '''lenEntrada = len(datasetEntrada[0][0]) if isRecurrent else len(datasetEntrada[0])
            # arrNNeuronios = [random.randint(128, 256) for _ in range(random.randint(1, 2))]
            if indexRede == 0:
                nNeuronsA = int((lenEntrada + len(datasetRotulo[0])) / 2)
                # nNeuronsB = int((nNeuronsA + len(datasetRotulo[0])) / 2)
            elif indexRede == 1:
                nNeuronsA = int((lenEntrada * 0.66)) + 2
                # nNeuronsB = int((nNeuronsA * 0.66)) + 2
            elif indexRede == 2:
                nNeuronsA = random.randint(lenEntrada, (lenEntrada * 3))
                # nNeuronsB = random.randint(nNeuronsA, (nNeuronsA * 2))
            else:'''
            if isDuplaHiposete:
                nNeuronsA = int(len(datasetEntrada[0][0])) + len(datasetRotulo[0])
                nNeuronsB = None  # int(len(datasetEntrada[0][0]) * 0.65) + len(datasetRotulo[0])
                nNeuronsC = None
            else:
                if isRecurrent:
                    nNeuronsA = int(len(datasetEntrada[0][0])) + len(datasetRotulo[0])
                    nNeuronsB = None
                else:
                    nNeuronsA = int(len(datasetEntrada[0])) + len(datasetRotulo[0])
                    nNeuronsB = None  # int(len(datasetEntrada[0][0]) * 0.65) + len(datasetRotulo[0])
                nNeuronsC = None

            arrNNeuronios = [nNeuronsA]
            if nNeuronsB is not None:
                arrNNeuronios.append(nNeuronsB)
            if nNeuronsC is not None:
                arrNNeuronios.append(nNeuronsC)
            modelSequencial = Sequential()

            for idxNNeuronios in range(len(arrNNeuronios)):
                returnSequences = idxNNeuronios < len(arrNNeuronios) - 1

                if isRecurrent:
                    if idxNNeuronios == 0:
                        inputShape = (len(datasetEntrada[0]), len(datasetEntrada[0][0]))
                        modelSequencial.add(SimpleRNN(units=arrNNeuronios[idxNNeuronios], activation="tanh",
                                                      input_shape=inputShape, return_sequences=returnSequences,
                                                      kernel_initializer=initializers.GlorotNormal(),
                                                      recurrent_initializer=initializers.GlorotNormal()))
                    else:
                        modelSequencial.add(SimpleRNN(units=arrNNeuronios[idxNNeuronios], activation="tanh",
                                                      return_sequences=returnSequences,
                                                      kernel_initializer=initializers.GlorotNormal(),
                                                      recurrent_initializer=initializers.GlorotNormal(),))
                else:
                    inputShape = (len(datasetEntrada[0]),)
                    if idxNNeuronios == 0:
                        modelSequencial.add(Dense(units=arrNNeuronios[idxNNeuronios], activation="tanh",
                                                  input_shape=inputShape,
                                                  kernel_initializer=initializers.GlorotNormal()))
                    else:
                        modelSequencial.add(Dense(units=arrNNeuronios[idxNNeuronios], activation="tanh",
                                                  kernel_initializer=initializers.GlorotNormal()))

            modelSequencial.add(Dense(units=len(datasetRotulo[0]), activation=funcAtiv,
                                      kernel_initializer=initializers.GlorotNormal()))
            arrRedesNeurais.append(modelSequencial)
        print(arrRedesNeurais[0].get_config())
        return arrRedesNeurais

    def getAndtrainRedesNeurais(self, datasetEntrada, datasetRotulo, datasetValidar,
                                funcAtiv: str = "softmax", nRedes: int = 10, isRecurrent: bool = True):
        arrRedesNeurais = self.buildRedesNeurais(datasetEntrada=datasetEntrada, datasetRotulo=datasetRotulo,
                                                 nRedes=nRedes, funcAtiv=funcAtiv, isRecurrent=isRecurrent)

        arrResultados = []
        qtdeDados = len(datasetEntrada)
        batchSize = int(qtdeDados * 0.25)
        qtdeDadosAvalidacao = 3  # int(len(datasetEntrada) * 0.33)

        for redeNeural in arrRedesNeurais:
            newDatasetEntrada = []
            newDatasetRotulo = []

            for indexRemove in range(qtdeDadosAvalidacao):
                entradaRemovido = datasetEntrada.pop()
                rotuloRemovido = datasetRotulo.pop()
                newDatasetEntrada.insert(0, entradaRemovido)
                newDatasetRotulo.insert(0, rotuloRemovido)

            funcStopping = EarlyStopping(monitor="accuracy", patience=250, mode="max")
            print(funcAtiv)
            funcLoss = "categorical_crossentropy" if funcAtiv == "softmax" else "binary_crossentropy"

            redeNeural.compile(optimizer=RMSprop(lr=1e-3), metrics=["accuracy"], loss=funcLoss)
            redeNeural.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRotulo), callbacks=[funcStopping],
                           epochs=2000, batch_size=batchSize, validation_data=datasetValidar)

            qtdeAcertos = [0, 0]
            for indexDadoVal in range(len(newDatasetEntrada)):
                previsaoValidar = numpy.array(redeNeural.predict(x=numpy.array([newDatasetEntrada[indexDadoVal]])))
                for dataValidar in previsaoValidar:
                    indexPrev = numpy.argmax(dataValidar)
                    indexRotu = numpy.argmax(newDatasetRotulo[indexDadoVal])
                    indexEmpate = int(len(datasetRotulo[0]) / 2)

                    if funcAtiv == "softmax":
                        if indexPrev == indexRotu:
                            qtdeAcertos[0] += 1
                        else:
                            qtdeAcertos[1] += 1
                    else:
                        for indx in range(len(dataValidar)):
                            valueNorm = 1 if dataValidar[indx] >= 0.5 else 0
                            if valueNorm == newDatasetRotulo[indexDadoVal][indx]:
                                qtdeAcertos[indx] += 1

                datasetEntrada.append(newDatasetEntrada[indexDadoVal])
                datasetRotulo.append(newDatasetRotulo[indexDadoVal])

                '''model.reset_states()
                model.reset_metrics()'''
                funcStopping = EarlyStopping(monitor="accuracy", patience=100, mode="max")
                redeNeural.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRotulo), callbacks=[funcStopping],
                               epochs=500, batch_size=batchSize)

            arrResultados.append(qtdeAcertos)

        indexMaiorAcerto = 0
        melhorArray = arrResultados[indexMaiorAcerto]
        for idxArrAcerto in range(len(arrResultados)):
            if funcAtiv == "softmax":
                if arrResultados[idxArrAcerto][0] >= melhorArray[0]:
                    melhorArray = arrResultados[idxArrAcerto]
                    indexMaiorAcerto = idxArrAcerto
            else:
                if sum(arrResultados[idxArrAcerto]) >= sum(melhorArray):
                    melhorArray = arrResultados[idxArrAcerto]
                    indexMaiorAcerto = idxArrAcerto

        bestRede = arrRedesNeurais[indexMaiorAcerto]
        '''bestRede.reset_states()
        bestRede.reset_metrics()
        funcStopping = EarlyStopping(monitor="accuracy", patience=200, mode="max")
        bestRede.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRotulo), callbacks=[funcStopping],
                     epochs=1000, batch_size=batchSize)'''

        return bestRede, melhorArray

    def getBestRedeNeuralByDataset(self, arrIdsTeam, isAgruparTeams, idTypeReturn,
                                   isFiltrarTeams, isRecurrent, funcAtiv, isPassadaTempoDupla,
                                   nRedes: int = 10, qtdeTentativas: int = 5, arrIdsExpecficos: List[int] = [],
                                   isDadosUmLadoSo: bool = True, idTypeRotulo: int = 1, idTypeEntrada: int = 1):
        idxBest = None
        datasetPrev = None
        bestQtdeDados = None
        bestLimitHistory = None
        listRedesNeurais = []
        listRedesNeuraisDuplaHipotese = []
        listArrayAcertos = []
        isDuplaHipotese = True
        for idxTentativa in range(qtdeTentativas):
            randomQtdeDados = random.randint(24, 24)
            randomQtdeDados = randomQtdeDados if randomQtdeDados % 2 == 0 else randomQtdeDados - 1
            randomLimitHistory = random.randint(5, 5)

            datasetEnt, datasetRot, datasetPrev, datasetValidar = self.obterDatasetRecurrentForKeras(
                arrIdsTeam=arrIdsTeam, isAgruparTeams=isAgruparTeams, idTypeReturn=idTypeReturn,
                isFiltrarTeams=isFiltrarTeams, isRecurrent=isRecurrent, funcAtiv=funcAtiv,
                isPassadaTempoDupla=isPassadaTempoDupla, qtdeDados=randomQtdeDados,
                limitHistoricoMedias=randomLimitHistory,  arrIdsExpecficos=arrIdsExpecficos,
                isDadosUmLadoSo=isDadosUmLadoSo, idTypeRotulo=idTypeRotulo, idTypeEntrada=idTypeEntrada)

            dateInicio = datetime.datetime.now()
            redeDuplaHipotese = None
            if isDuplaHipotese:
                redeNeural, arrAcertos, redeDuplaHipotese = self.getAndtrainRedesNeuraisEmDupla(
                    datasetEntrada=datasetEnt, datasetRotulo=datasetRot, datasetValidar=datasetValidar,
                    funcAtiv=funcAtiv, nRedes=nRedes, isRecurrent=isRecurrent, isDuplaHipotese=True)
            else:
                redeNeural, arrAcertos = self.getAndtrainRedesNeurais(datasetEntrada=datasetEnt, datasetRotulo=datasetRot,
                                                                      datasetValidar=datasetValidar, funcAtiv=funcAtiv,
                                                                      nRedes=nRedes, isRecurrent=isRecurrent)
            listRedesNeurais.append(redeNeural)
            listArrayAcertos.append(arrAcertos)

            if redeDuplaHipotese is not None:
                listRedesNeuraisDuplaHipotese.append(redeDuplaHipotese)

            dateFim = datetime.datetime.now()
            diference = (dateFim - dateInicio).total_seconds()

            msg = "\nTentativa " + str(idxTentativa) + "/" + str(qtdeTentativas)
            msg += " do(s) id(s): " + str(arrIdsTeam) + " levou " + str(diference) + " segundos."
            self.gravarLogs(msg)

            if idxBest is None:
                idxBest = idxTentativa
                bestQtdeDados = int(randomQtdeDados)
                bestLimitHistory = int(randomLimitHistory)
            else:
                if arrAcertos[0] > listArrayAcertos[-2][0]:
                    idxBest = idxTentativa
                    bestQtdeDados = int(randomQtdeDados)
                    bestLimitHistory = int(randomLimitHistory)

        if idxBest is None:
            raise Exception("Nenhuma rede treinada")

        msg = "\nCom assim: qtdeDados: " + str(bestQtdeDados) + ", randomLimite: " + str(bestLimitHistory) + "\n\n"
        msg += str(listRedesNeurais[idxBest].get_config()) + "\n"
        msg += "All acertos: " + str(listArrayAcertos) + "\n"
        msg += "Is dados 1 lado só: " + str(isDadosUmLadoSo)
        self.gravarLogs(msg=msg)
        predit = listRedesNeurais[idxBest].predict(x=numpy.array(datasetPrev))
        predit = numpy.array(predit).tolist()

        preditDupla = None
        if isDuplaHipotese:
            preditDupla = listRedesNeuraisDuplaHipotese[idxBest].predict(x=numpy.array(datasetPrev))
            preditDupla = numpy.array(preditDupla).tolist()

        return predit, listArrayAcertos[idxBest], preditDupla

    def gravarLogs(self, msg: str):
        self.ignoreIntelisense = True
        with open("C:/Users/lucas/OneDrive/Documentos/Projetos/footxap/web/static/js/resultados-dqn.txt",
                  "a", encoding="utf-8") as results:
            results.write(msg)
            results.close()

    def getAndtrainRedesNeuraisEmDupla(self, datasetEntrada, datasetRotulo, datasetValidar, funcAtiv: str = "softmax",
                                       nRedes: int = 10, isRecurrent: bool = True, isDuplaHipotese: bool = True):
        arrRedesNeurais = self.buildRedesNeurais(datasetEntrada=datasetEntrada, datasetRotulo=datasetRotulo,
                                                 nRedes=nRedes, funcAtiv=funcAtiv, isRecurrent=isRecurrent,
                                                 isDuplaHiposete=False)
        backupDatasetEntrada = deepcopy(datasetEntrada)
        arrResultados = []
        arrNewDatasetRotulosTwo = []
        qtdeDados = len(datasetEntrada)
        batchSize = int(qtdeDados * 0.25)
        qtdeDadosAvalidacao = len(datasetEntrada) - int(len(datasetEntrada) * 0.33)  # int(len(datasetEntrada) * 0.33)
        funcLoss = "categorical_crossentropy" if funcAtiv == "softmax" else "binary_crossentropy"
        arrPrevisoes = []

        for redeNeural in arrRedesNeurais:
            # cada indice representa nessa ordem
            # 1, X, 2, 1X, 12, X2
            # OR
            # 1 é o adversario e 2 o meu time prevendo
            exRotuloRedeTwo = [0, 0, 0, 0, 0, 0]
            newDatasetRotuloRedeTwo = []

            for idxVal in range(len(datasetEntrada) - qtdeDadosAvalidacao):
                indexPrev = numpy.argmax(datasetRotulo[idxVal])
                rotuloRedeTwo = list(exRotuloRedeTwo)
                rotuloRedeTwo[indexPrev] = 1
                newDatasetRotuloRedeTwo.append(rotuloRedeTwo)
                arrPrevisoes.append(datasetRotulo[idxVal])

            newDatasetEntrada = []
            newDatasetRotulo = []
            for indexRemove in range(qtdeDadosAvalidacao):
                entradaRemovido = datasetEntrada.pop()
                rotuloRemovido = datasetRotulo.pop()
                newDatasetEntrada.insert(0, entradaRemovido)
                newDatasetRotulo.insert(0, rotuloRemovido)

            funcStopping = EarlyStopping(monitor="accuracy", patience=250, mode="max")
            redeNeural.compile(optimizer=RMSprop(lr=1e-4, clipnorm=3), metrics=["accuracy"], loss=funcLoss)
            redeNeural.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRotulo), callbacks=[funcStopping],
                           epochs=2000, batch_size=batchSize, validation_data=datasetValidar)

            qtdeAcertos = [0, 0]
            for indexDadoVal in range(len(newDatasetEntrada)):
                previsaoValidar = numpy.array(redeNeural.predict(x=numpy.array([newDatasetEntrada[indexDadoVal]])))
                newRotuloTwo = list(exRotuloRedeTwo)

                for dataValidar in previsaoValidar:
                    if funcAtiv == "softmax":
                        indexPrev = numpy.argmax(dataValidar)
                        indexRotu = numpy.argmax(newDatasetRotulo[indexDadoVal])
                        indexEmpate = int(len(datasetRotulo[0]) / 2)
                        if indexPrev == indexRotu:
                            qtdeAcertos[0] += 1
                            newRotuloTwo[indexPrev] = 1
                        else:
                            if ((indexPrev == indexEmpate and indexRotu != indexEmpate) or
                                    (indexPrev != indexEmpate and indexRotu == indexEmpate)):
                                if indexPrev == 0 or indexRotu == 0:
                                    newRotuloTwo[3] = 1
                                elif indexPrev == 2 or indexRotu == 2:
                                    newRotuloTwo[5] = 1
                            else:
                                newRotuloTwo[4] = 1

                            qtdeAcertos[1] += 1
                    else:
                        for indx in range(len(dataValidar)):
                            valueNorm = 1 if dataValidar[indx] >= 0.5 else 0
                            if valueNorm == newDatasetRotulo[indexDadoVal][indx]:
                                qtdeAcertos[indx] += 1

                    arrPrevisoes.append(dataValidar)
                datasetEntrada.append(newDatasetEntrada[indexDadoVal])
                datasetRotulo.append(newDatasetRotulo[indexDadoVal])
                newDatasetRotuloRedeTwo.append(newRotuloTwo)

                '''model.reset_states()
                model.reset_metrics()'''
                funcStopping = EarlyStopping(monitor="accuracy", patience=100, mode="max")
                redeNeural.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRotulo), callbacks=[funcStopping],
                               epochs=500, batch_size=batchSize)

            arrResultados.append(qtdeAcertos)
            arrNewDatasetRotulosTwo.append(newDatasetRotuloRedeTwo)

            print(len(datasetRotulo))
            print(len(newDatasetRotuloRedeTwo))

            for i in range(len(datasetRotulo)):
                print(str(datasetRotulo[i]) + " == " + str(newDatasetRotuloRedeTwo[i]) + " == " + str(arrPrevisoes[i]))

        indexMaiorAcerto = 0
        melhorArray = arrResultados[indexMaiorAcerto]
        for idxArrAcerto in range(len(arrResultados)):
            if funcAtiv == "softmax":
                if arrResultados[idxArrAcerto][0] >= melhorArray[0]:
                    melhorArray = arrResultados[idxArrAcerto]
                    indexMaiorAcerto = idxArrAcerto
            else:
                if sum(arrResultados[idxArrAcerto]) >= sum(melhorArray):
                    melhorArray = arrResultados[idxArrAcerto]
                    indexMaiorAcerto = idxArrAcerto

        bestRede = arrRedesNeurais[indexMaiorAcerto]
        datasetRotuloDuplaHipotese = arrNewDatasetRotulosTwo[indexMaiorAcerto]

        redeDuplaHipotese = self.buildRedesNeurais(datasetEntrada=datasetEntrada,
                                                   datasetRotulo=datasetRotuloDuplaHipotese,
                                                   nRedes=1, funcAtiv=funcAtiv, isRecurrent=isRecurrent,
                                                   isDuplaHiposete=True)[0]

        funcStopping = EarlyStopping(monitor="accuracy", patience=250, mode="max")
        redeDuplaHipotese.compile(optimizer=RMSprop(lr=1e-4, clipnorm=5), metrics=["accuracy"], loss=funcLoss)
        redeDuplaHipotese.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRotuloDuplaHipotese),
                              callbacks=[funcStopping], epochs=1000, batch_size=batchSize)

        return bestRede, melhorArray, redeDuplaHipotese

    def getProbabilidadesWithSigmoid(self, arrIdsTeam: list, arrIdsExpecficos: list, isRecurrent: bool = True,
                                     funcAtiv: str = "sigmoid", isPassadaTempoDupla: bool = False,
                                     arrForRandomLimitHistoricoMedias: tuple = (3, 7), nRedes: int = 1,
                                     arrForRandomQtdeDados: tuple = (10, 18),
                                     isDadosUmLadoSo: bool = True, qtdeTentativas: int = 8, idTypeReturn: int = 1,
                                     idTypeRotulo: int = 1, idTypeEntrada: int = 2):
        msgInicio = "Parametros da função inicial: \n"
        msgInicio += "isRecurrent: {} / funcAtiv: {} / nRedes: {} / qtdeTentativas: {} / idTypeRotulo: {} / idTypeEntrada: {}\n".format(
            isRecurrent, funcAtiv, nRedes, qtdeTentativas, idTypeRotulo, idTypeEntrada)
        msgInicio += "isPassadaTempoDupla: {} / isDadosUmLadoSo:{}\n".format(isPassadaTempoDupla, isDadosUmLadoSo)
        msgInicio += "arrForRandomLimitHistoricoMedias: {} / arrForRandomQtdeDados: {}\n\n".format(
            arrForRandomLimitHistoricoMedias, arrForRandomQtdeDados)
        self.gravarLogs(msg=msgInicio)

        idVitoria = 1
        idEmpate = 2
        idDerrota = 3
        dateVitoriaInicio = datetime.datetime.now()
        self.gravarLogs("Treinando tentativa VITÓRIA: \n")
        previsaoVitoria, qtdeAcertosErrosVitoria = self.treinarBestTentativa(
            qtdeTentativas=qtdeTentativas, arrIdsTeam=arrIdsTeam, arrForRandomQtdeDados=arrForRandomQtdeDados,
            arrForRandomLimitHistoricoMedias=arrForRandomLimitHistoricoMedias, idTypeReturn=idVitoria,
            funcAtiv=funcAtiv, isRecurrent=isRecurrent, isDadosUmLadoSo=isDadosUmLadoSo,
            isPassadaTempoDupla=isPassadaTempoDupla, arrIdsExpecficos=arrIdsExpecficos, nRedes=nRedes,
            idTypeRotulo=idTypeRotulo, idTypeEntrada=idTypeEntrada
        )
        dateVitoriaFim = datetime.datetime.now()
        diference = (dateVitoriaFim - dateVitoriaInicio).total_seconds()
        msgVitoria = "Tentativa da VITÓRIA levou " + str(diference) + " segundos e ficou assim: \n"
        msgVitoria += str(previsaoVitoria) + " / " + str(qtdeAcertosErrosVitoria) + "\n\n"
        self.gravarLogs(msgVitoria)

        if (qtdeAcertosErrosVitoria[0] / sum(qtdeAcertosErrosVitoria) < 0.79 and
                qtdeAcertosErrosVitoria[1] / sum(qtdeAcertosErrosVitoria) < 0.79):
            msgErro = "Nao foi possivel prever acertou só {}".format(qtdeAcertosErrosVitoria)
            self.gravarLogs(msg=msgErro)
            return False

        dateEmpateInicio = datetime.datetime.now()
        self.gravarLogs("Treinando tentativa EMPATE: \n")
        previsaoEmpate, arrAcertosEmpate = 0, 0
        '''self.treinarBestTentativa(
            qtdeTentativas=qtdeTentativas, arrIdsTeam=arrIdsTeam, arrForRandomQtdeDados=arrForRandomQtdeDados,
            arrForRandomLimitHistoricoMedias=arrForRandomLimitHistoricoMedias, idTypeReturn=idEmpate,
            funcAtiv=funcAtiv, isRecurrent=isRecurrent, isDadosUmLadoSo=isDadosUmLadoSo,
            isPassadaTempoDupla=isPassadaTempoDupla, arrIdsExpecficos=arrIdsExpecficos, nRedes=nRedes,
            idTypeRotulo=idTypeRotulo
        )'''
        dateEmpateFim = datetime.datetime.now()
        diference = (dateEmpateFim - dateEmpateInicio).total_seconds()
        msgEmpate = "Tentativa da EMPATE levou " + str(diference) + " segundos e ficou assim: \n"
        msgEmpate += str(previsaoEmpate) + " / " + str(arrAcertosEmpate) + "\n\n"
        self.gravarLogs(msgEmpate)

        dateDerrotaInicio = datetime.datetime.now()
        self.gravarLogs("Treinando tentativa DERROTA: \n")
        previsaoDerrota, arrAcertosDerrota = 0, 0
        '''self.treinarBestTentativa(
            qtdeTentativas=qtdeTentativas, arrIdsTeam=arrIdsTeam, arrForRandomQtdeDados=arrForRandomQtdeDados,
            arrForRandomLimitHistoricoMedias=arrForRandomLimitHistoricoMedias, idTypeReturn=idDerrota,
            funcAtiv=funcAtiv, isRecurrent=isRecurrent, isDadosUmLadoSo=isDadosUmLadoSo,
            isPassadaTempoDupla=isPassadaTempoDupla, arrIdsExpecficos=arrIdsExpecficos, nRedes=nRedes
        )'''
        dateDerrotaFim = datetime.datetime.now()
        diference = (dateDerrotaFim - dateDerrotaInicio).total_seconds()
        msgDerrota = "Tentativa da DERROTA levou " + str(diference) + " segundos e ficou assim: \n"
        msgDerrota += str(previsaoDerrota) + " / " + str(arrAcertosDerrota) + "\n\n"
        self.gravarLogs(msgDerrota)

        msgFim = "Resultado final para aposta é esse time é: \n"
        msgFim += "Derrota / Empate / Vitória \n"
        msgFim += str([previsaoDerrota, previsaoEmpate, previsaoVitoria]) + " / " + str(qtdeAcertosErrosVitoria) + "\n\n"
        self.gravarLogs(msgFim)
        return msgFim

    def treinarBestTentativa(self, qtdeTentativas: int, arrIdsTeam: List[int], arrForRandomQtdeDados: tuple,
                             arrForRandomLimitHistoricoMedias: tuple, idTypeReturn: int, funcAtiv: str,
                             isRecurrent: bool, isDadosUmLadoSo: bool, isPassadaTempoDupla: bool, arrIdsExpecficos: list,
                             nRedes: int, idTypeRotulo: int, idTypeEntrada: int):
        idxBestEscolhido = None
        bestRedeEscolhida = None
        bestArrayAcertosEscolhida = None
        arrPrevisaoEscolhida = None
        lastBestArrayAcertos = None
        for idxTentativa in range(qtdeTentativas):
            dateTentativaInicio = datetime.datetime.now()
            randomQtdeDados = random.randint(arrForRandomQtdeDados[0], arrForRandomQtdeDados[1])
            randomQtdeDados = randomQtdeDados if randomQtdeDados % 2 == 0 else randomQtdeDados - 1
            randomLimitHistory = random.randint(arrForRandomLimitHistoricoMedias[0],
                                                arrForRandomLimitHistoricoMedias[1])

            datasetEnt, datasetRot, datasetPrev, datasetValidar = self.obterDatasetRecurrentForKeras(
                arrIdsTeam=arrIdsTeam, idTypeReturn=idTypeReturn,
                isRecurrent=isRecurrent, funcAtiv=funcAtiv,
                isPassadaTempoDupla=isPassadaTempoDupla, qtdeDados=randomQtdeDados,
                limitHistoricoMedias=randomLimitHistory, arrIdsExpecficos=arrIdsExpecficos,
                isDadosUmLadoSo=isDadosUmLadoSo, idTypeRotulo=idTypeRotulo, idTypeEntrada=idTypeEntrada)

            qtdeDadosValidar = 5  # int((len(datasetEnt) * 0.3))
            batchSize = 5  # int((len(datasetEnt) * 0.25))
            '''bestRede, bestArrayAcertos = self.treinarWithBestSigmoid(
                datasetEntrada=datasetEnt, datasetRotulo=datasetRot, nRedes=nRedes, funcAtiv=funcAtiv,
                isRecurrent=isRecurrent, qtdeDadosValidarDataset=qtdeDadosValidarDataset,
                batchSize=batchSize)'''

            bestRede, bestArrayAcertos = self.treinarWithBestSigmoid(
                datasetEntrada=datasetEnt, datasetRotulo=datasetRot, nRedes=nRedes, funcAtiv=funcAtiv,
                isRecurrent=isRecurrent, qtdeDadosValidar=qtdeDadosValidar,
                batchSize=batchSize)

            if bestRedeEscolhida is None:
                idxBestEscolhido = idxTentativa
                bestRedeEscolhida = bestRede
                bestArrayAcertosEscolhida = bestArrayAcertos
                arrPrevisaoEscolhida = bestRedeEscolhida.predict(x=numpy.array(datasetPrev))
                lastBestArrayAcertos = bestArrayAcertos
            elif ((bestArrayAcertos[0] > bestArrayAcertosEscolhida[0] and
                   bestArrayAcertos[0] >= bestArrayAcertosEscolhida[1]) or
                  (bestArrayAcertosEscolhida[0] < 3 and bestArrayAcertos[1] >= 4)):
                idxBestEscolhido = idxTentativa
                bestRedeEscolhida = bestRede
                bestArrayAcertosEscolhida = bestArrayAcertos
                arrPrevisaoEscolhida = bestRedeEscolhida.predict(x=numpy.array(datasetPrev))

            dateTentativaFim = datetime.datetime.now()
            diference = (dateTentativaFim - dateTentativaInicio).total_seconds()
            msg = "Tentativa {}/{} levou {} segundos. Com os seguintes parametros: ".format(
                idxTentativa, qtdeTentativas, diference)
            msg += "randomQtdeDados: {}, randomLimiHistory: {}, batchSize: {}, qtdeDadosValidarDataset: {}\n".format(
                randomQtdeDados, randomLimitHistory, batchSize, qtdeDadosValidar)
            self.gravarLogs(msg)

            if bestArrayAcertos[0] / sum(bestArrayAcertos) == 1:
                break

        if bestRedeEscolhida is None or bestArrayAcertosEscolhida is None:
            raise Exception("Nao achamos rede de uma conferida")
        msgEscolhido = "Foi escolhido a tentativa: {}\n".format(idxBestEscolhido)
        msgEscolhido += "Info REDE NEURAL ESCOLHIDA: {}".format(bestRedeEscolhida.get_config())
        msgEscolhido += "\n"
        self.gravarLogs(msgEscolhido)
        return numpy.array(arrPrevisaoEscolhida).tolist(), bestArrayAcertosEscolhida

    def treinarWithBestSigmoid(self, datasetEntrada: List, datasetRotulo: List, nRedes: int, funcAtiv: str,
                               isRecurrent: bool, qtdeDadosValidar: int, batchSize: int):
        if funcAtiv != "sigmoid" or len(datasetRotulo[0]) != 1:
            raise Exception("Param nao atendidos: " + funcAtiv + " / " + str(datasetRotulo[0]))

        arrRedesNeurais = self.buildRedesNeurais(datasetEntrada=datasetEntrada, datasetRotulo=datasetRotulo,
                                                 nRedes=nRedes, funcAtiv=funcAtiv, isRecurrent=isRecurrent,
                                                 isDuplaHiposete=False)
        funcLoss = "binary_crossentropy"
        arrArrAcertosErros = []

        for idxRedeNeural in range(len(arrRedesNeurais)):
            redeNeural = arrRedesNeurais[idxRedeNeural]
            bkpDatasetEntrada = deepcopy(datasetEntrada)
            bkpDatasetRotulo = deepcopy(datasetRotulo)

            dadosValidarEntrada = bkpDatasetEntrada[-qtdeDadosValidar:]
            dadosValidarRotulo = bkpDatasetRotulo[-qtdeDadosValidar:]

            dadosTreinarEntrada = bkpDatasetEntrada[:-qtdeDadosValidar]
            dadosTreinarRotulo = bkpDatasetRotulo[:-qtdeDadosValidar]

            funcStoppingA = EarlyStopping(monitor="accuracy", patience=750, mode="max")
            redeNeural.compile(optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"], loss=funcLoss)
            redeNeural.fit(x=numpy.array(dadosTreinarEntrada), y=numpy.array(dadosTreinarRotulo),
                           callbacks=[funcStoppingA], epochs=2000, batch_size=batchSize)

            qtdeAcertosErros = [0, 0]
            for idxDado in range(len(dadosValidarEntrada)):
                isValidarAcerto = 1  # random.randint(0, 1) if idxDado < len(dadosValidarEntrada) - 1 else 1

                if isValidarAcerto == 1:
                    arrPredict = redeNeural.predict(x=numpy.array([dadosValidarEntrada[idxDado]]))

                    for predict in arrPredict:
                        if len(predict) != 1:
                            raise Exception("UEPPAAAAA muitos dados pro predict" + str(predict) + str(idxDado))

                        thobleshot = 0.5
                        valuePredictNormalizado = 1 if predict[0] >= thobleshot else 0

                        if valuePredictNormalizado == dadosValidarRotulo[idxDado][0]:
                            qtdeAcertosErros[0] += 1
                        else:
                            qtdeAcertosErros[1] += 1

                dadosTreinarEntrada.append(dadosValidarEntrada[idxDado])
                dadosTreinarRotulo.append(dadosValidarRotulo[idxDado])

                if isValidarAcerto == 1:
                    funcStoppingB = EarlyStopping(monitor="accuracy", patience=300, mode="max")
                    redeNeural.fit(x=numpy.array(dadosTreinarEntrada), y=numpy.array(dadosTreinarRotulo),
                                   callbacks=[funcStoppingB], epochs=500, batch_size=batchSize)

            arrArrAcertosErros.append(qtdeAcertosErros)
            msg = "Rede {}/{} teve acertos: {}, qtdeDados: {}\n".format(
                idxRedeNeural, len(arrRedesNeurais), qtdeAcertosErros, len(datasetEntrada))
            self.gravarLogs(msg=msg)

        bestRede = arrRedesNeurais[0]
        bestQtdeAcertosErros = arrArrAcertosErros[0]

        for idx in range(len(arrArrAcertosErros)):
            if arrArrAcertosErros[idx][0] > bestQtdeAcertosErros[0]:
                bestQtdeAcertosErros = arrArrAcertosErros[idx]
                bestRede = arrRedesNeurais[idx]

        return bestRede, bestQtdeAcertosErros

