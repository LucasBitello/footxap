from __future__ import annotations

import os
import pickle
import random
import time

import numpy

from copy import deepcopy
from api.regras.iaFFRegras import FF, ModelDataFF
from api.regras.iaRNNRegras import RNN, ModelDataRNN
from api.regras.datasetPartidasRegras import DatasetPartidasRegras

class ModelPrevisao:
    def __init__(self):
        self.qtde_dados_entrada: int = None
        self.previsao: list[list[list]] = []
        self.arr_entradas_originais: list[list] = []
        self.arr_rotulos: list[list[list]] = []
        self.tx_aprendizado: float = None
        self.L2_regularizacao: float = None
        self.nEpocas: int = None
        self.data_previsao: str = None
        self.media_entropy: list = None
        self.media_accuracy: list = None
        self.msg_erro: str = None


class RedeLTSM:
    def preverComRNN(self, id_team_home: int, id_team_away: int = None, id_season: int = None, isPartida=False,
                     qtdeDados=55, isAmbas: bool = True, isGols: bool = True):

        arrIdsTeamSelec = [id_team_home]
        if isAmbas:
            arrIdsTeamSelec.append(id_team_away)

        resulRNN = [0, 0, 0]
        qtdeDadosA = 120 if isAmbas else 35
        qtdeDadosB = 130 if isAmbas else 45
        isAchouResultado = False
        for i in range(20):
            datasetPartidaRegras = DatasetPartidasRegras()
            datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(arrIdsTeam=arrIdsTeamSelec,
                                                                                      isNormalizarSaidaEmClasse=True,
                                                                                      isFiltrarTeams=True,
                                                                                      qtdeDados=random.randint(qtdeDadosA, qtdeDadosB),
                                                                                      isAgruparTeams=False,
                                                                                      isForFF=False,
                                                                                      isDadosSoUmLado=False,
                                                                                      isGols=isGols)
            isAchouResultado = False
            modelDataRNN = ModelDataRNN(arrEntradas=datasetEntrada, arrRotulos=datasetRotulo, arrDadosPrever=None,
                                        arrNameFuncAtivacaoCadaOculta=["tanh", "tanh", "tanh", "tanh"],
                                        arrNameFuncAtivacaoCadaSaida=["sigmoid" if isGols else "softmax", "sigmoid", "sigmoid", "sigmoid",
                                                                      "sigmoid", "sigmoid"])

            if len(datasetEntrada) < qtdeDadosA:
                raise Exception("Sem dado suficientes temos somente: " + str(len(datasetEntrada)))
            else:
                print("Vai ser somente com " + str(len(datasetEntrada)))

            arrDadosEntradas = modelDataRNN.iaRegras.obter_k_folds_temporal(datasetEntrada, 2)
            arrDadosRotulos = modelDataRNN.iaRegras.obter_k_folds_temporal(datasetRotulo, 2)

            entradaConcat = arrDadosEntradas[0]
            rotuloConcat = arrDadosRotulos[0]
            config = {}

            newModelData = deepcopy(modelDataRNN)
            newModelData.arr_n_camada_oculta = [15, 8] \
                if isGols else [15, 7]
            newModelData.taxa_aprendizado_culta = [random.uniform(0.01, 0.005) if isGols else random.uniform(0.001, 0.005),
                                                   random.uniform(0.005, 0.01) if isGols else random.uniform(0.005, 0.01),
                                                   random.uniform(0.005, 0.01)]
            newModelData.taxa_regularizacao_l2_oculta = [random.uniform(0.05, 0.1), random.uniform(0.01, 0.05),
                                                         random.uniform(0.001, 0.01)]

            newModelData.taxa_aprendizado_saida = [random.uniform(0.01, 0.05) if isGols else random.uniform(0.01, 0.05),
                                                   random.uniform(0.01, 0.05) if isGols else random.uniform(0.01, 0.05),
                                                   random.uniform(0.03, 0.05) if isGols else random.uniform(0.01, 0.05),
                                                   random.uniform(0.07, 0.01) if isGols else random.uniform(0.01, 0.05)]
            newModelData.taxa_regularizacao_l2_saida = [random.uniform(0.00001, 0.0001), random.uniform(0.00001, 0.0001),
                                                        random.uniform(0.00001, 0.0001), random.uniform(0.00001, 0.0001),
                                                        random.uniform(0.00001, 0.0001), random.uniform(0.00001, 0.0001),
                                                        random.uniform(0.00001, 0.0001), random.uniform(0.00001, 0.0001)]
            newModelData.n_epocas = 5000 if isGols else 5000
            newRede = RNN(modelDataRNN=newModelData)

            estados_ocultos = None
            for iEnt in range(len(arrDadosEntradas)):
                newRede.modelDataRNN.arr_dados_prever = datasetPrever
                newRede.modelDataRNN.arr_entradas = entradaConcat
                newRede.modelDataRNN.arr_rotulos = rotuloConcat
                resulRNN = newRede.treinar(
                    isBrekarPorEpocas=True, isAtualizarPesos=True,
                    qtdeDadoValidar=1 if iEnt < len(arrDadosEntradas) - 1 else 0,
                    estados_ocultos_anterior=estados_ocultos)

                for indexReg in range(len(newRede.txAprendizadoSaida)):
                    newRede.txAprendizadoSaida[indexReg] = (newRede.txAprendizadoSaida[indexReg] * 1.)
                    newRede.taxa_regularizacao_saida[indexReg] = (newRede.taxa_regularizacao_saida[indexReg] * 1)

                for indexReg in range(len(newRede.txAprendizadoOculta)):
                    newRede.txAprendizadoOculta[indexReg] = (newRede.txAprendizadoOculta[indexReg] * 1.)
                    newRede.taxa_regularizacao_oculta[indexReg] = (newRede.taxa_regularizacao_oculta[indexReg] * 1)

                newRede.nEpocas *= 2

                if resulRNN is not False:
                    estados_ocultos = resulRNN[3][-5:]

                    if iEnt < len(arrDadosEntradas) - 1:
                        """entradaConcat = numpy.concatenate((entradaConcat, arrDadosEntradas[iEnt + 1])).tolist()
                        rotuloConcat = numpy.concatenate((rotuloConcat, arrDadosRotulos[iEnt + 1])).tolist()"""
                        entradaConcat = arrDadosEntradas[iEnt + 1]
                        rotuloConcat = arrDadosRotulos[iEnt + 1]
                        for i in range(5):
                            aaa = arrDadosEntradas[iEnt].pop()
                            bbb = arrDadosRotulos[iEnt].pop()

                            entradaConcat.insert(0, aaa)
                            rotuloConcat.insert(0, bbb)

                    else:
                        isAchouResultado = True
                else:
                    break

            if isAchouResultado:
                break

        if not isAchouResultado:
            print("NÃO FOI POSSIVL ACHAR UM BOM RESULTADO")
            raise Exception("NÃO FOI POSSIVL ACHAR UM BOM RESULTADO")

        # arrConfigs = sorted(arrConfigs, key=lambda x: (sum(x["media_accuracy"])), reverse=False)
        return resulRNN[2]

    def preverComFF(self, id_team_home: int, id_team_away: int = None, id_season: int = None, isPartida=False,
                    qtdeDados=750, qtdeDadosValidar: int = 6, nFolds: int = 1, isAmbas: bool = True):

        arrIdsTeamSelec = [id_team_home]
        if isAmbas:
            arrIdsTeamSelec.append(id_team_away)

        resulFF = [0, 0, 0]
        qtdeDadosA = 120 if isAmbas else 60
        qtdeDadosB = 130 if isAmbas else 65
        for i in range(150):
            datasetPartidaRegras = DatasetPartidasRegras()
            datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(arrIdsTeam=arrIdsTeamSelec,
                                                                                      isNormalizarSaidaEmClasse=True,
                                                                                      isFiltrarTeams=True,
                                                                                      qtdeDados=random.randint(qtdeDadosA, qtdeDadosB),
                                                                                      isAgruparTeams=True,
                                                                                      isForFF=False,
                                                                                      isDadosSoUmLado=False)

            modelDataFF = ModelDataFF(arrEntradas=datasetEntrada, arrRotulos=datasetRotulo, arrDadosPrever=None,
                                      arrNameFuncAtivacaoCadaOculta=["tanh", "tanh", "tanh", "tanh"],
                                      arrNameFuncAtivacaoCadaSaida=["softmax", "sigmoid", "sigmoid", "sigmoid",
                                                                    "sigmoid", "sigmoid"])

            if len(datasetEntrada) < 60:
                raise Exception("Sem dado suficientes temos somente: " + str(len(datasetEntrada)))
            else:
                print("Vai ser somente com " + str(len(datasetEntrada)))
                time.sleep(3)

            isAchouResultado = False
            arrDadosEntradas = modelDataFF.iaRegras.obter_k_folds_temporal(datasetEntrada, 3)
            arrDadosRotulos = modelDataFF.iaRegras.obter_k_folds_temporal(datasetRotulo, 3)

            entradaConcat = arrDadosEntradas[0]
            rotuloConcat = arrDadosRotulos[0]
            config = {}

            newModelData = deepcopy(modelDataFF)
            newModelData.arr_n_camada_oculta = [18, 9]
            newModelData.taxa_aprendizado_culta = [0.01, 0.01, 0.001]
            newModelData.taxa_regularizacao_l2_oculta = [0.000005, 0.00000001, 0.0001]
            newModelData.taxa_aprendizado_saida = [0.005, 0.001, 0.001, 0.003, 0.005, 0.008]
            newModelData.taxa_regularizacao_l2_saida = [0.00000003, 0.00000025, 0.00000025, 0.000000004, 0.0001, 0.0008]
            newModelData.n_epocas = 2500
            newRede = FF(modelDataFF=newModelData)

            for iEnt in range(len(arrDadosEntradas)):
                newRede.modelDataFF.arr_dados_prever = datasetPrever
                newRede.modelDataFF.arr_entradas = entradaConcat
                newRede.modelDataFF.arr_rotulos = rotuloConcat
                resulFF = newRede.treinar(
                    isBrekarPorEpocas=True, isAtualizarPesos=True,
                    qtdeDadoValidar=1 if iEnt < len(arrDadosEntradas) - 1 else 0, n_folds=1)

                for indexReg in range(len(newRede.txAprendizadoSaida)):
                    newRede.txAprendizadoSaida[indexReg] = (newRede.txAprendizadoSaida[indexReg] * 0.5)
                    newRede.taxa_regularizacao_saida[indexReg] = (newRede.taxa_regularizacao_saida[indexReg] * 0.5)

                for indexReg in range(len(newRede.txAprendizadoOculta)):
                    newRede.txAprendizadoOculta[indexReg] = (newRede.txAprendizadoOculta[indexReg] * 0.5)
                    newRede.taxa_regularizacao_oculta[indexReg] = (newRede.taxa_regularizacao_oculta[indexReg] * 0.5)

                newRede.nEpocas *= 2

                if resulFF is not False:
                    if iEnt < len(arrDadosEntradas) - 1:
                        entradaConcat = numpy.concatenate((entradaConcat, arrDadosEntradas[iEnt + 1])).tolist()
                        rotuloConcat = numpy.concatenate((rotuloConcat, arrDadosRotulos[iEnt + 1])).tolist()
                    else:
                        isAchouResultado = True
                else:
                    break

            if isAchouResultado:
                break

        # arrConfigs = sorted(arrConfigs, key=lambda x: (sum(x["media_accuracy"])), reverse=False)
        return resulFF[2]

    @staticmethod
    def initArrayUniform(randA: int | float, randB: int | float, sizeArray: int, isEmbaralhar: bool = False) -> list:
        if type(randA) == float:
            array = numpy.linspace(randA, randB, sizeArray)
        else:
            passo = (randB - randA) // (sizeArray - 1)
            if passo == 0:
                if randA == randB:
                    array = numpy.asarray([randA for i in range(sizeArray)])
                else:
                    array = numpy.asarray(
                        [numpy.random.randint(randA if randA < randB else randB, randB if randB > randA else randA)
                         for i in range(sizeArray)])
            else:
                array = numpy.arange(randA, randB + 1, passo)
        array = array.tolist()

        if isEmbaralhar:
            numpy.random.shuffle(array)

        return array

    def gerarConfigs(self, idRede: int = 1, isAmbas: bool = False):
        arrLTSMs = []
        nRanges = 150

        randeNCamadasA = 1
        randeNCamadasB = 1

        randNeuroniosA = 22
        randNeuroniosB = 22

        randTxAprendizadoA = 0.02 if idRede == 1 else 0.05
        randTxAprendizadoB = 0.02 if idRede == 1 else 0.1

        randTxRegularizacaoA = 0.0001 if idRede == 1 else 0.005
        randTxRegularizacaoB = 0.0001 if idRede == 1 else 0.005

        arrayCamadas = self.initArrayUniform(randeNCamadasA, randeNCamadasB, nRanges, isEmbaralhar=True)
        arrayCamadasNormalizadas = []
        arrayNeuronios = self.initArrayUniform(randNeuroniosA, randNeuroniosB, nRanges, isEmbaralhar=True)
        arrayTxAprendizado = self.initArrayUniform(randTxAprendizadoA, randTxAprendizadoB, nRanges, isEmbaralhar=True)
        arrayTxRegulrizacao = self.initArrayUniform(randTxRegularizacaoA, randTxRegularizacaoB, nRanges, isEmbaralhar=True)

        for indexNCamada in range(len(arrayCamadas)):
            nCamadasNormalizada = []
            for j in range(arrayCamadas[indexNCamada]):
                nCamadasNormalizada.append(arrayNeuronios[indexNCamada])
            arrayCamadasNormalizadas.append(nCamadasNormalizada)

        for i in range(nRanges):
            idNCamadas = random.randint(1, 1)
            arrDictsLTSM = {
                "id": i,
                "taxa_aprendizado": arrayTxAprendizado[i],
                "taxa_regularizacao_l2": arrayTxRegulrizacao[i],
                "arr_n_camada_oculta": [30, 20, 10] if idRede == 1 else [30, 10],  # [20, 15, 5] ,
                "n_epocas": 10000
            }
            arrLTSMs.append(arrDictsLTSM)

        return arrLTSMs