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

    def obterDatasetRecurrentForKeras(self, arrIdsTeam: list = [], isAgruparTeams: bool = True, idTypeReturn: int = 1,
                                      isFiltrarTeams: bool = True, isRecurrent: bool = True, funcAtiv: str = "softmax",
                                      isPassadaTempoDupla: bool = True, qtdeDados: int = 40,
                                      limitHistoricoMedias: int = 5, arrIdsExpecficos: List[int] = [],
                                      isDadosUmLadoSo: bool =True):
        funcAtivacao = funcAtiv

        if len(arrIdsTeam) == 0:
            raise Exception("e preciso passar um id team pelo menos")

        isAntigoDataset = False
        if isAntigoDataset:
            datasetPartidaRegras = DatasetPartidasRegras()
            datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(
                arrIdsTeam=arrIdsTeam, isNormalizarSaidaEmClasse=funcAtivacao == "softmax",
                isFiltrarTeams=isFiltrarTeams, qtdeDados=qtdeDados, isAgruparTeams=isAgruparTeams,
                idTypeReturn=idTypeReturn, isPassadaTempoDupla=isPassadaTempoDupla,
                limitHistoricoMedias=limitHistoricoMedias)
        else:
            datasetRegra = DatasetRegras()
            datasetEntrada, datasetRotulo, datasetPrever = datasetRegra.obterDataset(arrIdsTeam=arrIdsTeam,
                                                                                     limitHistorico=limitHistoricoMedias,
                                                                                     qtdeDadosForTeam=qtdeDados,
                                                                                     arrIdsExpecficos=arrIdsExpecficos,
                                                                                     idTypeReturn=idTypeReturn,
                                                                                     isDadosUmLadoSo=isDadosUmLadoSo)

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
                          isRecurrent: bool = True):
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
            nNeuronsA = random.randint(51, 96)
            nNeuronsB = random.randint(96, 150)

            arrNNeuronios = [nNeuronsA]
            if nNeuronsB is not None:
                arrNNeuronios.append(nNeuronsB)
            modelSequencial = Sequential()

            for idxNNeuronios in range(len(arrNNeuronios)):
                returnSequences = idxNNeuronios < len(arrNNeuronios) - 1

                if isRecurrent:
                    if idxNNeuronios == 0:
                        inputShape = (len(datasetEntrada[0]), len(datasetEntrada[0][0]))
                        modelSequencial.add(SimpleRNN(units=arrNNeuronios[idxNNeuronios], activation="tanh",
                                                      input_shape=inputShape, return_sequences=returnSequences))
                    else:
                        modelSequencial.add(SimpleRNN(units=arrNNeuronios[idxNNeuronios], activation="tanh",
                                                      return_sequences=returnSequences))
                else:
                    inputShape = (len(datasetEntrada[0]),)
                    if idxNNeuronios == 0:
                        modelSequencial.add(Dense(units=arrNNeuronios[idxNNeuronios], activation="tanh",
                                                  input_shape=inputShape))
                    else:
                        modelSequencial.add(Dense(units=arrNNeuronios[idxNNeuronios], activation="tanh"))

            modelSequencial.add(Dense(units=len(datasetRotulo[0]), activation=funcAtiv))
            arrRedesNeurais.append(modelSequencial)

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
                                   isDadosUmLadoSo: bool = True):
        idxBest = None
        datasetPrev = None
        bestQtdeDados = None
        bestLimitHistory = None
        listRedesNeurais = []
        listArrayAcertos = []
        for idxTentativa in range(qtdeTentativas):
            randomQtdeDados = random.randint(25, 30)
            randomQtdeDados = randomQtdeDados if randomQtdeDados % 2 == 0 else randomQtdeDados - 1
            randomLimitHistory = random.randint(2, 3)

            datasetEnt, datasetRot, datasetPrev, datasetValidar = self.obterDatasetRecurrentForKeras(
                arrIdsTeam=arrIdsTeam, isAgruparTeams=isAgruparTeams, idTypeReturn=idTypeReturn,
                isFiltrarTeams=isFiltrarTeams, isRecurrent=isRecurrent, funcAtiv=funcAtiv,
                isPassadaTempoDupla=isPassadaTempoDupla, qtdeDados=randomQtdeDados,
                limitHistoricoMedias=randomLimitHistory,  arrIdsExpecficos=arrIdsExpecficos,
                isDadosUmLadoSo=isDadosUmLadoSo)

            dateInicio = datetime.datetime.now()
            redeNeural, arrAcertos = self.getAndtrainRedesNeurais(datasetEntrada=datasetEnt, datasetRotulo=datasetRot,
                                                                  datasetValidar=datasetValidar, funcAtiv=funcAtiv,
                                                                  nRedes=nRedes, isRecurrent=isRecurrent)
            listRedesNeurais.append(redeNeural)
            listArrayAcertos.append(arrAcertos)
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
        return predit, listArrayAcertos[idxBest]

    def gravarLogs(self, msg: str):
        self.ignoreIntelisense = True
        with open("C:/Users/lucas/OneDrive/Documentos/Projetos/footxap/web/static/js/resultados-dqn.txt",
                  "a", encoding="utf-8") as results:
            results.write(msg)
            results.close()

