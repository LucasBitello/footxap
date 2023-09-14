from __future__ import annotations

import os
import pickle
import random
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
                     qtdeDados=120, isAmbas: bool = False):

        arrIdsTeamSelec = [id_team_home]
        if isAmbas:
            arrIdsTeamSelec.append(id_team_away)

        datasetPartidaRegras = DatasetPartidasRegras()
        datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(arrIdsTeam=arrIdsTeamSelec,
                                                                                  isNormalizarSaidaEmClasse=False,
                                                                                  isFiltrarTeams=True,
                                                                                  qtdeDados=qtdeDados,
                                                                                  isAgruparTeams=True, isForFF=False,
                                                                                  isDadosSoUmLado=False)

        modelDataRNN = ModelDataRNN(arrEntradas=datasetEntrada, arrRotulos=datasetRotulo, arrDadosPrever=None,
                                    arrNameFuncAtivacaoCadaOculta=["tanh", "tanh", "tanh", "tanh"],
                                    arrNameFuncAtivacaoCadaSaida=["sigmoid", "sigmoid", "sigmoid", "sigmoid"])

        newModelData = None
        isAchouResultado = False
        arrConfigs = self.gerarConfigs(idRede=2)
        arrDadosEntradas = modelDataRNN.iaRegras.obter_k_folds_temporal(datasetEntrada, 2)
        arrDadosRotulos = modelDataRNN.iaRegras.obter_k_folds_temporal(datasetRotulo, 2)
        resulRNN = [0, 0, 0]

        for i in range(25):
            entradaConcat = arrDadosEntradas[0]
            rotuloConcat = arrDadosRotulos[0]
            config = {}

            newModelData = deepcopy(modelDataRNN)
            newModelData.arr_n_camada_oculta = [26, 10]
            newModelData.taxa_aprendizado = 0.05
            newModelData.taxa_regularizacao_l2 = 0.005
            newModelData.n_epocas = 15000
            newRede = RNN(modelDataRNN=newModelData)

            for iEnt in range(len(arrDadosEntradas)):
                newRede.modelDataRNN.arr_dados_prever = datasetPrever
                newRede.modelDataRNN.arr_entradas = entradaConcat
                newRede.modelDataRNN.arr_rotulos = rotuloConcat
                resulRNN = newRede.treinar(
                    isBrekarPorEpocas=True, isAtualizarPesos=True, qtdeDadoValidar=2)

                if resulRNN is not False:
                    if iEnt < len(arrDadosEntradas) - 1:
                        entradaConcat = numpy.concatenate((entradaConcat, arrDadosEntradas[iEnt + 1])).tolist()
                        rotuloConcat = numpy.concatenate((rotuloConcat, arrDadosRotulos[iEnt + 1])).tolist()
                    else:
                        resulRNN = newRede.treinar(
                            isBrekarPorEpocas=True, isAtualizarPesos=True, qtdeDadoValidar=0)
                        if resulRNN is not False:
                            isAchouResultado = True
                else:
                    break

            if isAchouResultado:
                break

        #arrConfigs = sorted(arrConfigs, key=lambda x: (sum(x["media_accuracy"])), reverse=False)
        return resulRNN[2]

    def preverComRNNBkp(self, id_team_home: int, id_team_away: int = None, id_season: int = None, isPartida=False,
                     qtdeDados=120, isAmbas: bool = False):

        arrIdsTeamSelec = [id_team_home]
        if isAmbas:
            arrIdsTeamSelec.append(id_team_away)

        datasetPartidaRegras = DatasetPartidasRegras()
        datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(arrIdsTeam=arrIdsTeamSelec,
                                                                                  isNormalizarSaidaEmClasse=False,
                                                                                  isFiltrarTeams=True,
                                                                                  qtdeDados=qtdeDados,
                                                                                  isAgruparTeams=True, isForFF=False,
                                                                                  isDadosSoUmLado=False)

        modelDataRNN = ModelDataRNN(arrEntradas=datasetEntrada, arrRotulos=datasetRotulo, arrDadosPrever=None,
                                    arrNameFuncAtivacaoCadaOculta=["tanh", "tanh", "tanh", "tanh"],
                                    arrNameFuncAtivacaoCadaSaida=["sigmoid", "sigmoid", "sigmoid", "sigmoid"])

        newModelData = None
        arrConfigs = self.gerarConfigs(idRede=2)
        arrDadosEntradasA = []
        arrDadosRotulosA = []
        arrDadosEntradasB = deepcopy(datasetEntrada)
        arrDadosRotulosB = deepcopy(datasetRotulo)

        for i in range(int(len(datasetEntrada) * 0.7)):
            arrayEntradaRemove = datasetEntrada.pop(0)
            arrayRotuloRemove = datasetRotulo.pop(0)
            arrDadosEntradasA.append(arrayEntradaRemove)
            arrDadosRotulosA.append(arrayRotuloRemove)

        for config in arrConfigs:
            newModelData = deepcopy(modelDataRNN)

            newModelData.arr_n_camada_oculta = config["arr_n_camada_oculta"]
            newModelData.taxa_aprendizado = config["taxa_aprendizado"]
            newModelData.taxa_regularizacao_l2 = config["taxa_regularizacao_l2"]
            newModelData.n_epocas = 1
            newRede = None

            with open(str(os.path.abspath("./api/regras/class_rnn.txt")), 'rb') as class_rnn:
                try:
                    newRede = pickle.load(class_rnn)
                    if len(newModelData.arr_entradas[0]) != len(newRede.modelDataRNN.arr_entradas[0]):
                        newRede = RNN(modelDataRNN=newModelData)
                        print("precisou criar novo arquivo RNN")
                    else:
                        newRede = RNN(modelDataRNN=newModelData)
                        # newRede = newRede.__init__(modelDataFF=newModelData, isNovosPesos=False)
                        print("Carregou arquivo")
                except EOFError:
                    newRede = RNN(modelDataRNN=newModelData)
                    print("não carregou arquivo RNN")

            #rede = paramEscolhidos["rede"]
            newRede.nEpocas = 5
            newRede.modelDataRNN.arr_dados_prever = datasetPrever
            newRede.modelDataRNN.arr_entradas = arrDadosEntradasA
            newRede.modelDataRNN.arr_rotulos = arrDadosRotulosA
            config["media_entropy"], config["media_accuracy"], config["sdf"] = newRede.treinar(isBrekarPorEpocas=True, isAtualizarPesos=False, qtdeDadoValidar=0)

            config["rede"] = deepcopy(newRede)

        arrConfigs = sorted(arrConfigs, key=lambda x: (sum(x["media_accuracy"])), reverse=False)

        for i in arrConfigs:
            print(i)

        resulRNN = None
        for i in range(15):
            paramEscolhidos = arrConfigs[i]

            newRede = paramEscolhidos["rede"]
            newRede.nEpocas = 15000
            newRede.modelDataRNN.arr_dados_prever = datasetPrever
            resulRNN = newRede.treinar(isBrekarPorEpocas=True, isAtualizarPesos=True, qtdeDadoValidar=1)

            if resulRNN is not False:
                newRede.modelDataRNN.arr_entradas = arrDadosEntradasB
                newRede.modelDataRNN.arr_rotulos = arrDadosRotulosB
                newRede.nEpocas = 20000
                resulRNN = newRede.treinar(isBrekarPorEpocas=True, isAtualizarPesos=True, qtdeDadoValidar=2)

                if resulRNN is not False:
                    resulRNN = newRede.treinar(isBrekarPorEpocas=True, isAtualizarPesos=True, qtdeDadoValidar=0)

                    if resulRNN is not False:
                        break


        return resulRNN[2]

    def preverComFF(self, id_team_home: int, id_team_away: int = None, id_season: int = None, isPartida=False,
                    qtdeDados=750, qtdeDadosValidar: int = 6, nFolds: int = 1):

        datasetPartidaRegras = DatasetPartidasRegras()
        datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(arrIdsTeam=[id_team_home, id_team_away],
                                                                                  isNormalizarSaidaEmClasse=False,
                                                                                  isFiltrarTeams=True,
                                                                                  qtdeDados=qtdeDados,
                                                                                  isAgruparTeams=False, isForFF=True,
                                                                                  isDadosSoUmLado=False)

        modelDataFF = ModelDataFF(arrEntradas=datasetEntrada, arrRotulos=datasetRotulo, arrDadosPrever=None,
                                  arrNameFuncAtivacaoCadaOculta=["tanh", "tanh", "tanh", "tanh", "tanh", "tanh"],
                                  arrNameFuncAtivacaoCadaSaida=["sigmoid", "sigmoid", "sigmoid", "sigmoid"])

        newModelData = None
        arrConfigs = self.gerarConfigs(idRede=1)

        for config in arrConfigs:
            newModelData = deepcopy(modelDataFF)
            newModelData.arr_n_camada_oculta = config["arr_n_camada_oculta"]
            newModelData.taxa_aprendizado = config["taxa_aprendizado"]
            newModelData.taxa_regularizacao_l2 = config["taxa_regularizacao_l2"]
            newModelData.n_epocas = 1
            newRede = None

            with open(str(os.path.abspath("./api/regras/class_ff.txt")), 'rb') as class_ff:
                try:
                    newRede = pickle.load(class_ff)
                    if len(newModelData.arr_entradas[0]) != len(newRede.modelDataFF.arr_entradas[0]):
                        newRede = FF(modelDataFF=newModelData)
                        print("precisou criar novo arquivo FF")
                    else:
                        newRede = FF(modelDataFF=newModelData)
                        # newRede = newRede.__init__(modelDataFF=newModelData, isNovosPesos=False)
                        print("Carregou arquivo")
                except EOFError:
                    newRede = FF(modelDataFF=newModelData)
                    print("não carregou arquivo FF")

            """newRede.nEpocas = 1
            config["media_entropy"], config["media_accuracy"], resul = newRede.treinar(
                isBrekarPorEpocas=True, isAtualizarPesos=False, qtdeDadoValidar=qtdeDadosValidar, n_folds=nFolds)
            config["rede"] = deepcopy(newRede)"""

            rede = newRede
            rede.nEpocas = 10000
            rede.modelDataFF.arr_dados_prever = datasetPrever
            resulFF = rede.treinar(isBrekarPorEpocas=True, isAtualizarPesos=True,
                                   qtdeDadoValidar=qtdeDadosValidar, n_folds=nFolds)

            if resulFF is not False:
                with open("./api/regras/class_ff.txt", 'wb') as class_ff:
                    pickle.dump(rede, class_ff)

                resulFF = rede.treinar(isBrekarPorEpocas=True, isAtualizarPesos=True,
                                       qtdeDadoValidar=qtdeDadosValidar, n_folds=nFolds)

                resulFF = rede.treinar(isBrekarPorEpocas=True, isAtualizarPesos=True,
                                       qtdeDadoValidar=0, n_folds=nFolds, isForcarTreino=True)

                if resulFF is not False:
                    break
            else:
                print("is criou novos arquivos")
                with open("./api/regras/class_ff.txt", 'wb') as class_ff:
                    pickle.dump(FF(modelDataFF=newModelData, isNovosPesos=True), class_ff)

        return resulFF[2]

        arrConfigs = sorted(arrConfigs, key=lambda x: (x["media_entropy"][0]), reverse=False)

        for i in arrConfigs:
            print(i)

        resulFF = False
        for i in range(15):
            paramEscolhidos = arrConfigs[i]

            rede = paramEscolhidos["rede"]
            rede.nEpocas = 7000
            rede.modelDataFF.arr_dados_prever = datasetPrever
            resulFF = rede.treinar(isBrekarPorEpocas=True, isAtualizarPesos=True,
                                   qtdeDadoValidar=qtdeDadosValidar, n_folds=nFolds)

            if resulFF is not False:
                with open("./api/regras/class_ff.txt", 'wb') as class_ff:
                    pickle.dump(rede, class_ff)
                break
            else:
                print("is criou novos arquivos")
                with open("./api/regras/class_ff.txt", 'wb') as class_ff:
                    pickle.dump(FF(modelDataFF=newModelData, isNovosPesos=True), class_ff)

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
        randTxAprendizadoB = 0.02 if idRede == 1 else 0.05

        randTxRegularizacaoA = 0.0001 if idRede == 1 else 0.05
        randTxRegularizacaoB = 0.0001 if idRede == 1 else 0.05

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