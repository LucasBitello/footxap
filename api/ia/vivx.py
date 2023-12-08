import random
import time

from api.networks.recurrent import NetworkRecurrentMultiple, ParamsRecurrentMultiple
from api.datasets.datasetPartida import DatasetPartida


class Vivx:

    def treinarVivxByTeamOnBatch(self, arrIdTeam: list[int], qtdeDadosMedia: int = 5):
        datasetPartida = DatasetPartida()

        datasets = datasetPartida.obterDatasets(arrIdsTeams=arrIdTeam, historicoMedias=qtdeDadosMedia)
        msg = ""
        for dataset in datasets:
            if dataset.is_prever:
                paramsRecurrentMultiple = ParamsRecurrentMultiple()
                paramsRecurrentMultiple.datasetEntrada = dataset.dataset_entrada
                paramsRecurrentMultiple.datasetRotulo = dataset.dataset_rotulo
                paramsRecurrentMultiple.datasetPrever = dataset.dataset_prever
                paramsRecurrentMultiple.nomeTipoRede = "Rede do team " + dataset.name_team
                paramsRecurrentMultiple.qtdeNeuroniosEntrada = len(dataset.dataset_entrada[0][0][0])
                paramsRecurrentMultiple.shapeEntrada = (len(dataset.dataset_entrada[0][0]),
                                                        len(dataset.dataset_entrada[0][0][0]))
                paramsRecurrentMultiple.learningRate = 1e-3
                paramsRecurrentMultiple.arrQtdeNeuroniosOculta = [10]
                paramsRecurrentMultiple.arrNomesAtivacaoOculta = ["tanh", "tanh", "tanh"]

                paramsRecurrentMultiple.arrQtdeNeuroniosSaida = []

                for idxOut in range(len(dataset.dataset_rotulo[0])):
                    paramsRecurrentMultiple.arrQtdeNeuroniosSaida.append(len(dataset.dataset_rotulo[0][idxOut][0]))
                    paramsRecurrentMultiple.arrNomesAtivacaoSaida.append("softmax")
                    paramsRecurrentMultiple.arrNameLosses.append("categorical_crossentropy")
                    # paramsRecurrentMultiple.arrNameLosses.append("binary_crossentropy")

                paramsRecurrentMultiple.batchSize = len(dataset.dataset_entrada)
                paramsRecurrentMultiple.qtdeDadosValidar = 0
                paramsRecurrentMultiple.nEpocas = 250

                network = NetworkRecurrentMultiple()
                returnPrev = network.trainOnBatchBKP(params=paramsRecurrentMultiple)
                nTent = 0
                while not returnPrev:
                    if nTent >= 5:
                        print("Nao foi possicel achar prev para " + dataset.name_team)
                        returnPrev = ([], [], "Nao foi possicel achar prev para " + dataset.name_team)
                        time.sleep(5)
                        break
                    returnPrev = network.trainOnBatchBKP(params=paramsRecurrentMultiple)
                    paramsRecurrentMultiple.arrQtdeNeuroniosOculta[0] += 10
                    nTent += 1
                model, arrPrev, msgR = returnPrev
                msg += dataset.name_team + "\n"
                msg += msgR + "\n"

                msaB = "--------------------------------------------\n"
                msaB += dataset.name_team + "\n"
                msaB += msgR + "\n"
                self.gravarLogsB(msaB)

        return msg

    @staticmethod
    def gravarLogs(msg: str):
        path = open('C:/Users/lucas/OneDrive/Documentos/Projetos/footxap/web/static/js/results.txt', 'at',
                    encoding="utf-8")
        with path as results:
            results.write(msg)
        path.close()

    @staticmethod
    def gravarLogsB(msg: str):
        path = open('C:/Users/lucas/OneDrive/Documentos/Projetos/footxap/web/static/js/resultsB.txt', 'at',
                    encoding="utf-8")
        with path as results:
            results.write(msg)
        path.close()
