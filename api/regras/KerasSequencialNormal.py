import datetime

import numpy
import random
import tensorflow
from copy import deepcopy

from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, SimpleRNN, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.losses import MSE

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
                                      limitHistoricoMedias: int = 5):
        funcAtivacao = funcAtiv

        if len(arrIdsTeam) == 0:
            raise Exception("e preciso passar um id team pelo menos")

        datasetPartidaRegras = DatasetPartidasRegras()
        datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(
            arrIdsTeam=arrIdsTeam, isNormalizarSaidaEmClasse=funcAtivacao == "softmax",
            isFiltrarTeams=isFiltrarTeams, qtdeDados=qtdeDados, isAgruparTeams=isAgruparTeams,
            idTypeReturn=idTypeReturn, isPassadaTempoDupla=isPassadaTempoDupla,
            limitHistoricoMedias=limitHistoricoMedias)

        if len(datasetEntrada) < qtdeDados:
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

                    arrRotulos.append(rotulos[i][0])

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

    def buildRedesNeurais(self, datasetEntrada, datasetRotulo, nRedes: int = 10, funcAtiv: str = "softmax"):
        self.ignoreIntelisense = True
        arrRedesNeurais = []

        for indexRede in range(nRedes):
            arrNNeuronios = [random.randint(len(datasetEntrada[0]), 128) for _ in range(random.randint(1, 1))]
            modelSequencial = Sequential()
            inputShape = (len(datasetEntrada[0]), len(datasetEntrada[0][0]))

            for idxNNeuronios in range(len(arrNNeuronios)):
                returnSequences = idxNNeuronios < len(arrNNeuronios) - 1

                if idxNNeuronios == 0:
                    modelSequencial.add(SimpleRNN(units=arrNNeuronios[idxNNeuronios], activation="tanh",
                                                  input_shape=inputShape, return_sequences=returnSequences))
                else:
                    modelSequencial.add(SimpleRNN(units=arrNNeuronios[idxNNeuronios], activation="tanh",
                                                  return_sequences=returnSequences))

            modelSequencial.add(Dense(units=len(datasetRotulo[0]), activation=funcAtiv))
            arrRedesNeurais.append(modelSequencial)

        return arrRedesNeurais

    def getAndtrainRedesNeurais(self, datasetEntrada, datasetRotulo, datasetValidar,
                                funcAtiv: str = "softmax", nRedes: int = 10):
        arrRedesNeurais = self.buildRedesNeurais(datasetEntrada=datasetEntrada, datasetRotulo=datasetRotulo,
                                                 nRedes=nRedes, funcAtiv=funcAtiv)

        arrResultados = []
        qtdeDadosAvalidacao = 5

        qtdeDados = len(datasetEntrada)
        for redeNeural in arrRedesNeurais:
            newDatasetEntrada = []
            newDatasetRotulo = []

            for indexRemove in range(qtdeDadosAvalidacao):
                entradaRemovido = datasetEntrada.pop()
                rotuloRemovido = datasetRotulo.pop()
                newDatasetEntrada.insert(0, entradaRemovido)
                newDatasetRotulo.insert(0, rotuloRemovido)

            funcStopping = EarlyStopping(monitor="accuracy", patience=250, mode="max")
            funcLoss = "categorical_crossentropy" if funcAtiv == "softmax" else "binary_crossentropy"

            redeNeural.compile(optimizer=Adam(learning_rate=0.001), metrics=["accuracy"], loss=funcLoss)
            redeNeural.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRotulo), callbacks=[funcStopping],
                           epochs=1000, batch_size=int(qtdeDados * 0.25), validation_data=datasetValidar)

            qtdeAcertos = [0, 0, 0]
            for indexDadoVal in range(len(newDatasetEntrada)):
                previsaoValidar = numpy.array(redeNeural.predict(x=numpy.array([newDatasetEntrada[indexDadoVal]]))[0])
                indexPrev = numpy.argmax(previsaoValidar)
                indexRotu = numpy.argmax(newDatasetRotulo[indexDadoVal])
                indexEmpate = int(len(datasetRotulo[0]) / 2)

                if (indexPrev == indexRotu or
                        ((indexPrev < indexEmpate and indexRotu < indexEmpate) or
                         (indexPrev > indexEmpate and indexRotu > indexEmpate))):
                    qtdeAcertos[0] += 1
                elif ((indexPrev == indexEmpate and indexRotu != indexEmpate) or
                      (indexPrev != indexEmpate and indexRotu == indexEmpate)):
                    qtdeAcertos[1] += 1
                else:
                    qtdeAcertos[2] += 1

                datasetEntrada.append(newDatasetEntrada[indexDadoVal])
                datasetRotulo.append(newDatasetRotulo[indexDadoVal])

                '''model.reset_states()
                model.reset_metrics()'''
                funcStopping = EarlyStopping(monitor="accuracy", patience=75, mode="max")
                redeNeural.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRotulo), callbacks=[funcStopping],
                               epochs=500, batch_size=int(qtdeDados * 0.25))

            arrResultados.append(qtdeAcertos)

        indexMaiorAcerto = random.randint(0, len(arrRedesNeurais) - 1)
        melhorArray = arrResultados[indexMaiorAcerto]
        for idxArrAcerto in range(len(arrResultados)):
            if arrResultados[idxArrAcerto][0] >= melhorArray[0]:
                melhorArray = arrResultados[idxArrAcerto]
                indexMaiorAcerto = idxArrAcerto

        return arrRedesNeurais[indexMaiorAcerto], melhorArray

    def getBestRedeNeuralByDataset(self, arrIdsTeam, isAgruparTeams, idTypeReturn,
                                   isFiltrarTeams, isRecurrent, funcAtiv, isPassadaTempoDupla,
                                   nRedes: int = 10, qtdeTentativas: int = 5):
        bestRedeNeural = None
        bestArrayAcertos = None
        datasetPrev = None
        for idxTentativa in range(qtdeTentativas):
            randomQtdeDados = random.randint(30, 60) if isPassadaTempoDupla else random.randint(10, 15)
            randomQtdeDados = randomQtdeDados if randomQtdeDados % 2 == 0 else randomQtdeDados - 1
            randomLimitHistory = idxTentativa + 1 #random.randint(2, 5)

            datasetEnt, datasetRot, datasetPrev, datasetValidar = self.obterDatasetRecurrentForKeras(
                arrIdsTeam=arrIdsTeam, isAgruparTeams=isAgruparTeams, idTypeReturn=idTypeReturn,
                isFiltrarTeams=isFiltrarTeams, isRecurrent=isRecurrent, funcAtiv=funcAtiv,
                isPassadaTempoDupla=isPassadaTempoDupla, qtdeDados=randomQtdeDados,
                limitHistoricoMedias=randomLimitHistory)

            dateInicio = datetime.datetime.now()
            redeNeural, arrAcertos = self.getAndtrainRedesNeurais(datasetEntrada=datasetEnt, datasetRotulo=datasetRot,
                                                                  datasetValidar=datasetValidar, funcAtiv=funcAtiv,
                                                                  nRedes=nRedes)
            dateFim = datetime.datetime.now()
            diference = (dateFim - dateInicio).total_seconds()

            msg = "\nTentativa " + str(idxTentativa) + "/" + str(qtdeTentativas)
            msg += " do(s) id(s): " + str(arrIdsTeam) + " levou " + str(diference) + " segundos."
            self.gravarLogs(msg)

            if bestRedeNeural is None and bestArrayAcertos is None:
                bestRedeNeural = redeNeural
                bestArrayAcertos = arrAcertos
            else:
                if arrAcertos[0] > bestArrayAcertos[0]:
                    bestRedeNeural = redeNeural
                    bestArrayAcertos = arrAcertos

        if bestRedeNeural is None:
            raise Exception("Nenhuma rede treinada")

        predit = bestRedeNeural.predict(x=numpy.array(datasetPrev))
        predit = numpy.array(predit).tolist()
        return predit, bestArrayAcertos

    def gravarLogs(self, msg: str):
        self.ignoreIntelisense = True
        with open("C:/Users/lucas/OneDrive/Documentos/Projetos/footxap/web/static/js/resultados-dqn.txt",
                  "a", encoding="utf-8") as results:
            results.write(msg)
            results.close()

