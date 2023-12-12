from datetime import datetime

import numpy

from api.datasets.datasetPartida import DatasetPartida
from matplotlib import pyplot as plt
import mplcursors


class Graficos:
    def mostrarGrafcos(self, arrIdTeam: list[int], qtdeDadosMedia: int = 5):
        datasetPartida = DatasetPartida()
        arrDados = datasetPartida.obterDatasets(arrIdTeam, qtdeDadosMedia)
        arrKeysX = []
        arrKeysY = []
        arrKeysYGoals = []
        nameTeams = ""
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
        yticks_interval = 1
        for idxDado, dado in enumerate(arrDados):
            if dado.id_team in arrIdTeam:
                nameTeams += dado.name_team + " / "
                print(dado.name_team)
                arrX = []
                arrEscanterios = []
                arrGolsMarcados = []
                arrGolsSofridos = []
                for idxEnt, ent in enumerate(dado.arr_obj_for_dataFrame):
                    if idxEnt >= len(dado.arr_obj_for_dataFrame) - 1:
                        print("lastDate: " + dado.arr_obj_for_dataFrame[idxEnt - 1].date.strftime("%Y-%m-%d %H:%M:%S"))
                        break

                    arrX.append(idxEnt)

                    if ent.corner_kicks is None:
                        arrEscanterios.append(-1)
                    else:
                        arrEscanterios.append(ent.corner_kicks)

                    if ent.goals_fulltime is None:
                        arrGolsMarcados.append(-1)
                    else:
                        arrGolsMarcados.append(ent.goals_fulltime)

                    if ent.goals_fulltime_conceded is None:
                        arrGolsSofridos.append(-1)
                    else:
                        arrGolsSofridos.append(ent.goals_fulltime_conceded)

                arrKeysX.append(arrX)
                if dado.arr_obj_for_dataFrame[-1].is_home:
                    ax1.plot(arrX, arrGolsMarcados, label="Gols Marcados")
                    ax1.plot(arrX, arrGolsSofridos, label="Gols Sofridos")
                    ax1.set_title("Gols " + dado.name_team)
                    ax1.legend(loc='upper left')
                    ax1.grid(True, linestyle='--', linewidth=0.3, color='gold')

                else:
                    ax2.plot(arrX, arrGolsMarcados, label="Gols Marcados")
                    ax2.plot(arrX, arrGolsSofridos, label="Gols Sofridos")
                    ax2.set_title("Gols " + dado.name_team)
                    ax2.legend(loc='upper left')
                    ax2.grid(True, linestyle='--', linewidth=0.3, color='lightgray')

                ax3.plot(arrX, arrEscanterios, label="Escanteios " + dado.name_team)
                ax3.legend(loc='upper left')
                ax3.set_title(nameTeams)
                ax3.grid(True, linestyle='--', linewidth=0.3, color='lightgray')

        plt.grid(True, linestyle='--', linewidth=0.3, color='gold')
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.4)
        plt.show()

    def mostrarGrafcosB(self, arrIdTeam: list[int], qtdeDadosMedia: int = 5):
        datasetPartida = DatasetPartida()
        arrDados = datasetPartida.obterDatasets(arrIdTeam, qtdeDadosMedia)
        arrKeysX = []
        arrKeysY = []
        arrKeysYGoals = []
        nameTeams = ""
        for idxDado, dado in enumerate(arrDados):
            if dado.id_team in arrIdTeam:
                nameTeams += dado.name_team + " / "
                print(dado.name_team)
                arrX = []
                arrY = []
                arrYGoals = []
                for idxEnt, ent in enumerate(dado.arrDictsMediaEntrada):
                    if idxEnt >= len(dado.arr_obj_for_dataFrame) - 1:
                        break

                    arrX.append(idxEnt)
                    if ent["m"+str(dado.historico_medias) + "_corner_kicks"] is None:
                        arrY.append(-1)
                    else:
                        arrY.append(ent["m"+str(dado.historico_medias) + "_corner_kicks"])

                    if ent["m"+str(dado.historico_medias) + "_goals_fulltime"] is None:
                        arrYGoals.append(-1)
                    else:
                        arrYGoals.append(ent["m"+str(dado.historico_medias) + "_goals_fulltime"])

                arrKeysX.append(arrX)
                arrKeysY.append(arrY)
                arrKeysYGoals.append(arrYGoals)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        for idxArr, arr in enumerate(arrKeysY):
            if idxArr == 0:
                ax1.plot(arrKeysX[0], arr, label="escanteios {}".format(idxArr))
            else:
                ax1.scatter(arrKeysX[0], arr, label="escanteios {}".format(idxArr))
            ax1.legend()
            ax1.set_title(nameTeams)
            ax2.plot(arrKeysX[0], arrKeysYGoals[idxArr], label="gols {}".format(idxArr))
            ax2.legend()
            ax2.set_title(nameTeams)
        plt.show()
