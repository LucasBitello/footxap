from __future__ import annotations

import random
import numpy
from copy import deepcopy
from api.regras.iaUteisRegras import IAUteisRegras
from api.regras.statisticsRegras import StatisticsRegras
from api.regras.iaLTSMRegras import LSTM, ModelDataLTSM
from api.regras.datasetPartidasRegras import DatasetPartidasRegras
from api.regras.iaFFRegras import FF, ModelDataFF
from api.regras.iaRNNRegras import RNN, ModelDataRNN

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
    def preverComLTSM(self, id_team_home: int, id_team_away: int = None, id_season: int = None, isPartida=False,
                      qtdeDados=25) -> ModelPrevisao:
        iaRegras = IAUteisRegras()
        modelFF = ModelDataFF(arrEntradas=[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                              arrRotulos=[[[0], [1]], [[0], [1]], [[1], [0]], [[1], [0]]],
                              arrDadosPrever=[[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

        modelLSTM = ModelDataLTSM(arrEntradas=[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                                arrRotulos=[[[0]], [[1]], [[0]], [[1]]],
                                arrDadosPrever=[[0, 0, 0, 0, 1]], arrNameFuncAtivacaoCadaSaida=["sigmoid"])


        modelRNN = ModelDataRNN(arrEntradas=[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                              arrRotulos=[[[0]], [[1]], [[0]], [[1]], [[1]]],
                              arrDadosPrever=[[0, 0, 0, 0, 1]], arrNameFuncAtivacaoCadaSaida=["softmax"])

        iaFF = FF(modelDataFF=modelFF)
        iaLTSM = LSTM(modelDataLTSM=modelLSTM)
        iaRNN = RNN(modelDataRNN=modelRNN)
        #iaFF.treinar()
        #iaLTSM.treinar()
        #iaRNN.treinar()

        isLSTM = False
        """statisticsRegras = StatisticsRegras()
        dataset = statisticsRegras.obterDatasetNormalizadoTeamsPlays(id_team_home=id_team_home,
                                                                     id_team_away=id_team_away, id_season=id_season,
                                                                     isPartida=isPartida, qtdeDados=qtdeDados)

        arrRotulosNormalizados = iaRegras.normalizarRotulosEmClasses(arrRotulosOriginais=dataset.arr_dados_rotulos_original,
                                                                     max_value_rotulos=dataset.max_value_rotulos)

        modelDataFF = ModelDataLTSM(arrEntradas=dataset.arr_dados_entrada, arrRotulos=arrRotulosNormalizados,
                                      arrRotulosOriginais=dataset.arr_dados_rotulos_original,
                                      arrDadosPrever=None, arrNameFuncAtivacaoCadaSaida=["softmax"])"""

        datasetPartidaRegras = DatasetPartidasRegras()
        datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(
            arrIdsTeam=[id_team_home, id_team_away], isNormalizarSaidaEmClasse=True, isFiltrarTeams=True)

        modelDataFF = ModelDataFF(arrEntradas=datasetEntrada, arrRotulos=datasetRotulo, arrDadosPrever=None,
                                  arrNameFuncAtivacaoCadaSaida=["sigmoid", "sigmoid", "softmax"],
                                  arrNameFuncAtivacaoCadaOculta=["sigmoid", "sigmoid", "sigmoid"])

        modelDataLSTM = ModelDataLTSM(arrEntradas=datasetEntrada, arrRotulos=datasetRotulo, arrDadosPrever=None,
                                      arrNameFuncAtivacaoCadaSaida=["softmax", "sigmoid", "softmax"])

        modelDataRNN = ModelDataRNN(arrEntradas=datasetEntrada, arrRotulos=datasetRotulo, arrDadosPrever=None,
                                    arrNameFuncAtivacaoCadaOculta=["sigmoid", "sigmoid", "sigmoid"],
                                    arrNameFuncAtivacaoCadaSaida=["sigmoid", "sigmoid", "softmax"])

        arrConfigs = self.gerarConfigs(isLSTM=isLSTM)

        for config in arrConfigs:
            if isLSTM:
                newModelData = deepcopy(modelDataLSTM)
            else:
                newModelData = deepcopy(modelDataRNN)
            newModelData.arr_n_camada_oculta = config["arr_n_camada_oculta"]
            newModelData.taxa_aprendizado = config["taxa_aprendizado"]
            newModelData.taxa_regularização_l2 = config["taxa_regularização_l2"]
            newModelData.n_epocas = 1

            newRede = LSTM(modelDataLTSM=newModelData) if isLSTM else RNN(modelDataRNN=newModelData)
            config["media_entropy"], config["media_accuracy"] = newRede.treinar(isBrekarPorEpocas=True,
                                                                                isAtualizarPesos=False)

            config["rede"] = deepcopy(newRede)

        arrConfigs = sorted(arrConfigs, key=lambda x: x["media_accuracy"][0], reverse=True)

        for i in arrConfigs:
            print(i)

        bestInit = None

        for i in range(10):
            paramEscolhidos = arrConfigs[i]
            newModel = deepcopy(modelDataLSTM) if isLSTM else deepcopy(modelDataRNN)
            newModel.n_epocas = 10000
            newModel.arr_n_camada_oculta = paramEscolhidos["arr_n_camada_oculta"]
            newModel.taxa_regularização_l2 = paramEscolhidos["taxa_regularização_l2"]
            newModel.taxa_aprendizado = paramEscolhidos["taxa_aprendizado"]

            rede = paramEscolhidos["rede"]
            rede.nEpocas = 13000
            if isLSTM:
                rede.modelDataLTSM.arr_dados_prever = datasetPrever
            else:
                rede.modelDataRNN.arr_dados_prever = datasetPrever
            ssss = rede.treinar(isBrekarPorEpocas=True, isAtualizarPesos=True)

            if ssss is not False:
                print(ssss)
                break

        if not isLSTM:
            previsao = bestInit.prever(entradas=datasetPrever, isPrintar=True)

        newModelPrevisao = ModelPrevisao()
        newModelPrevisao.previsao = previsao
        newModelPrevisao.L2_regularizacao = modelDataFF.taxa_regularização_l2
        newModelPrevisao.tx_aprendizado = modelDataFF.taxa_aprendizado
        newModelPrevisao.arr_entradas_originais = dataset.arr_dados_entrada_original
        newModelPrevisao.arr_rotulos = modelDataFF.arr_rotulos
        newModelPrevisao.data_previsao = dataset.data_previsao
        newModelPrevisao.qtde_dados_entrada = len(dataset.arr_dados_entrada_original)
        newModelPrevisao.media_entropy = media_entropy
        newModelPrevisao.media_accuracy = media_accuracy
        newModelPrevisao.msg_erro = ""

        for index in range(len(media_entropy)):
            newModelPrevisao.msg_erro += f"Entropia camd {index}: {media_entropy[index]:.2f}% \n"
            newModelPrevisao.msg_erro += f"Acuracia camd {index}: {media_accuracy[index] * 100:.2f}% \n "
            if index >= 1:
                newModelPrevisao.msg_erro += "\n"

        newModelPrevisao.previsao[0].append(newModelPrevisao.previsao[0][0])
        return newModelPrevisao


    def initArrayUniform(self, randA: int|float, randB: int|float, sizeArray: int, isEmbaralhar: bool = False):
        if type(randA) == float:
            array = numpy.linspace(randA, randB, sizeArray)
        else:
            passo = (randB - randA) // (sizeArray - 1)
            if passo == 0:
                if randA == randB:
                    array = numpy.asarray([randA for i in range(sizeArray)])
                else:
                    array = numpy.asarray([numpy.random.randint(randA if randA < randB else randB, randB if randB > randA else randA) for i in range(sizeArray)])
            else:
                array = numpy.arange(randA, randB + 1, passo)
        array = array.tolist()

        if isEmbaralhar:
            numpy.random.shuffle(array)
        return array


    def gerarConfigs(self, isLSTM: bool):
        arrLTSMs = []
        nRanges = 150

        randeNCamadasA = 1 if isLSTM else 1
        randeNCamadasB = 1 if isLSTM else 1

        randNeuroniosA = 22 if isLSTM else 22
        randNeuroniosB = 22 if isLSTM else 22

        randTxAprendizadoA = 0.1 if isLSTM else 0.1
        randTxAprendizadoB = 0.1 if isLSTM else 0.1

        randTxRegularizacaoA = 0.001 if isLSTM else 0.01
        randTxRegularizacaoB = 0.001 if isLSTM else 0.01


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
            arrDictsLTSM = {
                "id": i,
                "taxa_aprendizado": arrayTxAprendizado[i],
                "taxa_regularização_l2": arrayTxRegulrizacao[i],
                "arr_n_camada_oculta": [20, 18, 10],
                "n_epocas": 5000
            }
            arrLTSMs.append(arrDictsLTSM)

        return arrLTSMs