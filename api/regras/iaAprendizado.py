import random
from copy import deepcopy
from api.regras.iaRegras import IARegras
from api.regras.statisticsRegras import StatisticsRegras
from api.regras.iaLTSMRegras import LSTM, ModelDataLTSM

class ModelPrevisao:
    def __init__(self):
        self.qtde_dados_entrada = int = None
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
        iaRegras = IARegras()
        statisticsRegras = StatisticsRegras()
        dataset = statisticsRegras.obterDatasetNormalizadoTeamsPlays(id_team_home=id_team_home,
                                                                     id_team_away=id_team_away, id_season=id_season,
                                                                     isPartida=isPartida, qtdeDados=qtdeDados)

        arrRotulosNormalizados = iaRegras.normalizarRotulosEmClasses(arrRotulosOriginais=dataset.arr_dados_rotulos_original,
                                                                     max_value_rotulos=dataset.max_value_rotulos)


        modelDataLTSM = ModelDataLTSM(arrEntradas=dataset.arr_dados_entrada, arrRotulos=arrRotulosNormalizados,
                                      arrRotulosOriginais=dataset.arr_dados_rotulos_original,
                                      arrDadosPrever=None, arrNameFuncAtivacaoCadaSaida=["softmax"])


        arrLTSMs = []
        for i in range(5):
            arrDictsLTSM = {
                "id": i
            }

            if isPartida:
                nroCamadasOcultas = random.randint(1, 1)
                randA = 150 + qtdeDados
                randB = 400 + qtdeDados
                arrDictsLTSM["taxa_aprendizado"] = random.uniform(0.001, 0.0001)
                arrDictsLTSM["taxa_regularização_l2"] = random.uniform(0.01, 0.001)
            else:
                nroCamadasOcultas = random.randint(1, 1)
                randA = 150 + qtdeDados
                randB = 300 + qtdeDados
                arrDictsLTSM["taxa_aprendizado"] = random.uniform(0.001, 0.0001)
                arrDictsLTSM["taxa_regularização_l2"] = random.uniform(0.01, 0.001)

            arr_n_camadas = []
            for j in range(nroCamadasOcultas):
                if j >= 1:
                    randA = int(randA)
                    randB = int(randB)

                arr_n_camadas.append(random.randint(randA, randB))

            arrDictsLTSM["arr_n_camada_oculta"] = arr_n_camadas


            newModelData = deepcopy(modelDataLTSM)
            newModelData.arr_n_camada_oculta = arrDictsLTSM["arr_n_camada_oculta"]
            newModelData.taxa_aprendizado = arrDictsLTSM["taxa_aprendizado"]
            newModelData.taxa_regularização_l2 = arrDictsLTSM["taxa_regularização_l2"]
            newModelData.n_epocas = 100 if not isPartida else 150

            newLTSM = LSTM(modelDataLTSM=newModelData)
            arrDictsLTSM["media_entropy"], arrDictsLTSM["media_accuracy"] = newLTSM.treinar()

            arrLTSMs.append(arrDictsLTSM)

        arrLTSMs = sorted(arrLTSMs, key=lambda x: sum(x["media_entropy"]), reverse=False)

        for i in arrLTSMs:
            print(i)

        paramEscolhidos = arrLTSMs[0]
        modelDataLTSM.n_epocas = 350
        modelDataLTSM.arr_n_camada_oculta = paramEscolhidos["arr_n_camada_oculta"]
        modelDataLTSM.taxa_regularização_l2 = paramEscolhidos["taxa_regularização_l2"]
        modelDataLTSM.taxa_aprendizado = paramEscolhidos["taxa_aprendizado"]

        lstm = LSTM(modelDataLTSM=modelDataLTSM)
        media_entropy, media_accuracy = lstm.treinar()

        previsao = lstm.prever(entradas=dataset.arr_dados_prever, isPrintar=True)

        newModelPrevisao = ModelPrevisao()
        newModelPrevisao.previsao = previsao
        newModelPrevisao.L2_regularizacao = modelDataLTSM.taxa_regularização_l2
        newModelPrevisao.tx_aprendizado = modelDataLTSM.taxa_aprendizado
        newModelPrevisao.arr_entradas_originais = dataset.arr_dados_entrada_original
        newModelPrevisao.arr_rotulos = modelDataLTSM.arr_rotulos
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


        return newModelPrevisao