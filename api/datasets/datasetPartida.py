from __future__ import annotations

import numpy
import pandas
import warnings

from enum import Enum
from copy import deepcopy
from pandas import DataFrame
from datetime import datetime
from api.models.teamsModel import TeamsModel, Team
from api.models.fixturesTeamsModel import FixtureTeams
from api.models.fixturesModel import FixturesModel, Fixture
from api.models.fixturesTeamsStatisticsModel import FixtureTeamStatistic


class EnumTypeStatistics(Enum):
    shots_on_goal: int = 1
    shots_off_goal: int = 2
    shots_total: int = 3
    shots_bloqued: int = 4
    shots_insidebox: int = 5
    shots_outsidebox: int = 6
    fouls: int = 7
    corner_kicks: int = 8
    offsides: int = 9
    ball_possession: float = 10
    cards_yellow: int = 11
    cards_red: int = 12
    goalkeeper_saves: int = 13
    passes_total: int = 14
    passes_accurate: int = 15
    passes_precision: int = 16
    expected_goals: int = 17


class DatasetTeam:
    def __init__(self):
        self.id_team: str | None = None
        self.name_team: str | None = None
        self.is_possui_next_game: bool | None = None
        self.date_next_game: datetime | None = None
        self.historico_medias: int | None = None
        self.len_dataframe: int = 0
        self.dataFrame: DataFrame = DataFrame()
        self.arr_obj_for_dataFrame: list[ObjForDataFrame] = []

        self.arr_ids_adversarios: list[int] = []
        self.is_prever: int = 0
        self.minimos_valores: list = []
        self.amplitudes: list = []
        self.dataset_entrada: list = []
        self.dataset_rotulo: list = []
        self.dataset_prever: list = []

        self.arrDictsMediaEntrada: list[dict] = []
        self.arrDictsMediaRotulo: list[dict] = []


class ObjForDataFrame:
    def __init__(self):
        self.id_fixture: int | None = None
        self.id_season: int | None = None
        self.date: datetime | None = None
        self.time_elapsed: int | None = None
        self.status: str | None = None
        self.has_statistics_fixture: int | None = None

        self.id_team: int | None = None
        self.id_team_adversario: int | None = None
        self.is_winner: int | None = None
        self.is_vitoria: int | None = None
        self.is_empate: int | None = None
        self.is_derrota: int | None = None
        self.is_home: int | None = None

        self.goals: int | None = None
        self.goals_halftime: int | None = None
        self.goals_fulltime: int | None = None
        self.goals_extratime: int | None = None
        self.goals_penalty: int | None = None

        self.goals_conceded: int | None = None
        self.goals_halftime_conceded: int | None = None
        self.goals_fulltime_conceded: int | None = None
        self.goals_extratime_conceded: int | None = None
        self.goals_penalty_conceded: int | None = None

        # Infos da tabela fixture_teams_statistics
        self.shots_on_goal: int | None = None
        self.shots_off_goal: int | None = None
        self.shots_total: int | None = None
        self.shots_bloqued: int | None = None
        self.shots_insidebox: int | None = None
        self.shots_outsidebox: int | None = None
        self.fouls: int | None = None
        self.corner_kicks: int | None = None
        self.offsides: int | None = None
        self.ball_possession: int | None = None
        self.cards_yellow: int | None = None
        self.cards_red: int | None = None
        self.goalkeeper_saves: int | None = None
        self.passes_total: int | None = None
        self.passes_accurate: int | None = None
        self.passes_precision: int | None = None
        self.expected_goals: int | None = None

    @staticmethod
    def arrKeysMedia():
        return [
            "goals", "goals_halftime", "goals_fulltime", "is_winner", "is_vitoria", "is_empate", "is_derrota",

            "goals_conceded", "goals_halftime_conceded", "goals_fulltime_conceded",

            "shots_on_goal", "shots_off_goal", "shots_total", "shots_bloqued", "shots_insidebox",
            "shots_outsidebox", "fouls", "corner_kicks", "offsides", "ball_possession", "cards_yellow",
            "cards_red", "goalkeeper_saves", "passes_total", "passes_accurate", "passes_precision",
        ]

    @staticmethod
    def arrKeysRotulo():
        return [
            "goals_fulltime", "goals_fulltime_conceded"
        ]

    @staticmethod
    def arrKeysMediaIgnorar():
        return []

    @staticmethod
    def arrKeysIgnorarDataset():
        return [
            "id_team", "date", "time_elapsed", "status", "has_statistics_fixture",

            "is_winner", "is_vitoria", "is_empate", "is_derrota",

            "goals",  "goals_halftime",  "goals_fulltime", "goals_extratime",
            "goals_penalty",

            "goals_conceded", "goals_halftime_conceded", "goals_fulltime_conceded", "goals_extratime_conceded",
            "goals_penalty_conceded",

            "shots_on_goal", "shots_off_goal", "shots_total", "shots_bloqued", "shots_insidebox", "shots_outsidebox",
            "fouls", "corner_kicks", "offsides", "ball_possession", "cards_yellow", "cards_red", "goalkeeper_saves",
            "passes_total", "passes_accurate", "passes_precision", "expected_goals"
        ]

    def getDict(self):
        return self.__dict__


class DatasetPartida:
    def __init__(self):
        pandas.options.display.max_columns = None
        pandas.options.display.max_rows = None
        pandas.options.display.width = 2000
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.limit_dados: int = 10

    def obterDatasets(self, arrIdsTeams: list[int], historicoMedias: int):
        arrDatasetTeam: list[DatasetTeam] = []
        arrAllIdsTeam: list[int] = list(arrIdsTeams)

        idxIdTeam = 0
        while idxIdTeam < len(arrAllIdsTeam):
            datasetTeam = self.obterDatasetTeam(idTeam=arrAllIdsTeam[idxIdTeam], historicoMedias=historicoMedias)

            if datasetTeam.id_team in arrIdsTeams:
                datasetTeam.is_prever = 1

            if arrAllIdsTeam[idxIdTeam] in arrIdsTeams:
                for idAdversario in datasetTeam.arr_ids_adversarios:
                    if idAdversario not in arrAllIdsTeam:
                        arrAllIdsTeam.append(idAdversario)
            arrDatasetTeam.append(datasetTeam)
            idxIdTeam += 1

        arrDatasetTeam = self.obterDictMediasDatasetTeamsA(arrDatasetTeams=arrDatasetTeam)
        arrDatasetTeam = self.obterDatasetTrainG(arrDatasetTeam=arrDatasetTeam)
        return arrDatasetTeam

    def obterDatasetTeam(self, idTeam: int, historicoMedias: int):
        fixtureModel = FixturesModel(teamsModel=TeamsModel())
        fixtureModel.atualizarDados(arr_ids_team=[idTeam])

        team: Team = fixtureModel.teamsModel.obterByColumnsID(arrDados=[idTeam])[0]
        arrFixtures: list[Fixture] = fixtureModel.obterFixturesOrderDataBy(
            id_team=team.id, limit=self.limit_dados + historicoMedias, isASC=False)

        arrFixtures = arrFixtures[::-1]
        arrNextFixtures: list[Fixture] = fixtureModel.obterNextFixtureByidSeasonTeam(id_team=team.id)

        datasetTeam = DatasetTeam()
        datasetTeam.id_team = team.id
        datasetTeam.name_team = team.name
        datasetTeam.dataFrame = DataFrame(columns=numpy.array(ObjForDataFrame().__dict__.keys()))
        datasetTeam.arr_ids_adversarios = []

        if len(arrNextFixtures) >= 1:
            for nextFixture in arrNextFixtures:
                arrFixtures.append(nextFixture)
                datasetTeam.date_next_game = nextFixture.date
                datasetTeam.is_possui_next_game = True
        else:
            datasetTeam.is_possui_next_game = False

        for fixture in arrFixtures:
            arrFixturesTeam: list[FixtureTeams] = fixtureModel.fixturesTeamsModel.obterByColumns(
                arrNameColuns=["id_fixture"], arrDados=[fixture.id])

            newObjForDataFrame = ObjForDataFrame()
            for fixtureTeam in arrFixturesTeam:
                if fixtureTeam.id_team == team.id:
                    newObjForDataFrame.id_fixture = fixture.id
                    newObjForDataFrame.id_season = fixture.id_season
                    newObjForDataFrame.date = fixture.date
                    newObjForDataFrame.time_elapsed = fixture.time_elapsed
                    newObjForDataFrame.status = fixture.status
                    newObjForDataFrame.has_statistics_fixture = fixture.has_statistics_fixture

                    newObjForDataFrame.id_team = fixtureTeam.id_team
                    newObjForDataFrame.is_winner = 2 if fixtureTeam.is_winner == 1 else 1 \
                        if fixtureTeam.is_winner is None else 0
                    newObjForDataFrame.is_vitoria = int(fixtureTeam.is_winner == 1)
                    newObjForDataFrame.is_empate = int(fixtureTeam.is_winner is None)
                    newObjForDataFrame.is_derrota = int(fixtureTeam.is_winner == 0)
                    newObjForDataFrame.is_home = fixtureTeam.is_home

                    newObjForDataFrame.goals = fixtureTeam.goals
                    newObjForDataFrame.goals_halftime = fixtureTeam.goals_halftime
                    newObjForDataFrame.goals_fulltime = fixtureTeam.goals_fulltime
                    newObjForDataFrame.goals_extratime = fixtureTeam.goals_extratime
                    newObjForDataFrame.goals_penalty = fixtureTeam.goals_penalty

                    newObjForDataFrame.goals_conceded = fixtureTeam.goals_conceded
                    newObjForDataFrame.goals_halftime_conceded = fixtureTeam.goals_halftime_conceded
                    newObjForDataFrame.goals_fulltime_conceded = fixtureTeam.goals_fulltime_conceded
                    newObjForDataFrame.goals_extratime_conceded = fixtureTeam.goals_extratime_conceded
                    newObjForDataFrame.goals_penalty_conceded = fixtureTeam.goals_penalty_conceded

                    if fixture.has_statistics_fixture == 1:
                        arrStatistics: list[FixtureTeamStatistic] = (
                            fixtureModel.fixturesTeamsStatisticsModel.obterByColumns(
                                arrNameColuns=["id_fixture", "id_team"], arrDados=[fixture.id, team.id]))

                        if len(arrStatistics) == 0:
                            raise Exception(
                                "Diz que tem statistics mas nao tem, verifique id fixture " + str(fixture.id) +
                            " id_team: " + str(team.id))

                        for statistic in arrStatistics:
                            if statistic.id_type_statistic == EnumTypeStatistics.shots_on_goal.value:
                                newObjForDataFrame.shots_on_goal = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.shots_off_goal.value:
                                newObjForDataFrame.shots_off_goal = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.shots_total.value:
                                newObjForDataFrame.shots_total = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.shots_bloqued.value:
                                newObjForDataFrame.shots_bloqued = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.fouls.value:
                                newObjForDataFrame.fouls = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.corner_kicks.value:
                                newObjForDataFrame.corner_kicks = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.offsides.value:
                                newObjForDataFrame.offsides = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.ball_possession.value:
                                newObjForDataFrame.ball_possession = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.cards_yellow.value:
                                newObjForDataFrame.cards_yellow = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.cards_red.value:
                                newObjForDataFrame.cards_red = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.goalkeeper_saves.value:
                                newObjForDataFrame.goalkeeper_saves = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.passes_total.value:
                                newObjForDataFrame.passes_total = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.passes_accurate.value:
                                newObjForDataFrame.passes_accurate = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.passes_precision.value:
                                newObjForDataFrame.passes_precision = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.shots_insidebox.value:
                                newObjForDataFrame.shots_insidebox = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.shots_outsidebox.value:
                                newObjForDataFrame.shots_insidebox = statistic.value
                            elif statistic.id_type_statistic == EnumTypeStatistics.expected_goals.value:
                                newObjForDataFrame.shots_insidebox = statistic.value
                            else:
                                raise Exception("Parece q temos uma nova statistica nao cadastrada " +
                                                str(fixture.id) + " " + str(statistic.id))
                else:
                    newObjForDataFrame.id_team_adversario = fixtureTeam.id_team

                    if fixtureTeam.id_team not in datasetTeam.arr_ids_adversarios:
                        datasetTeam.arr_ids_adversarios.append(fixtureTeam.id_team)

            if len(arrFixturesTeam) == 2:
                datasetTeam.arr_obj_for_dataFrame.append(newObjForDataFrame)
                datasetTeam.dataFrame.loc[len(datasetTeam.dataFrame)] = newObjForDataFrame.__dict__

        datasetTeam.historico_medias = historicoMedias
        datasetTeam.len_dataframe = len(datasetTeam.dataFrame)
        return datasetTeam

    def obterDictMediasDatasetTeamsA(self, arrDatasetTeams: list[DatasetTeam]):
        for idxDatasetTeam in range(len(arrDatasetTeams)):
            datasetTeam = arrDatasetTeams[idxDatasetTeam]
            datasetTeam.arrDictsMediaEntrada, datasetTeam.arrDictsMediaRotulo = self.getDictMediasDatasetsA(
                datasetTeam=datasetTeam)

        return arrDatasetTeams

    @staticmethod
    def getDictMediasDatasetsA(datasetTeam: DatasetTeam):
        dictEntrada = dict()
        dictRotulo = dict()
        arrDictMediasEntradas = []
        arrDictMediasRotulos = []
        idxRangeA = 0
        idxRangeB = datasetTeam.historico_medias
        arrObjsForDataFrame = datasetTeam.arr_obj_for_dataFrame

        for idxObjDataFrame in range(len(arrObjsForDataFrame)):
            objDataFrame = arrObjsForDataFrame[idxObjDataFrame]
            dictEntrada["is_home"] = objDataFrame.is_home
            dictEntrada["date"] = int(objDataFrame.date.timestamp() * 1000)
            dictEntrada["id_fixture"] = objDataFrame.id_fixture
            dictEntrada["id_team"] = objDataFrame.id_team
            dictEntrada["id_team_adversario"] = objDataFrame.id_team_adversario

            for key in ObjForDataFrame().arrKeysMedia():
                nameKeyDict = "m" + str(datasetTeam.historico_medias) + "_" + key
                if idxObjDataFrame < datasetTeam.historico_medias:
                    dictEntrada[nameKeyDict] = -1
                else:
                    sumValuesKey = 0
                    arrObjsDataFrame = arrObjsForDataFrame[idxRangeA:idxRangeB]

                    for idxObjDataFrameOfArray in range(len(arrObjsDataFrame)):
                        objDataFrameOfArray = arrObjsDataFrame[idxObjDataFrameOfArray]
                        valueKey = objDataFrameOfArray.getDict()[key]
                        if valueKey is None:
                            valueKey = arrObjsForDataFrame[idxRangeB - 2].getDict()[key]
                            if valueKey is None:
                                valueKey = arrObjsForDataFrame[idxRangeB - 3].getDict()[key]
                                if valueKey is None:
                                    valueKey = -1
                        sumValuesKey += valueKey

                    mediaValuesKey = sumValuesKey / len(arrObjsDataFrame)
                    dictEntrada[nameKeyDict] = mediaValuesKey

            if idxObjDataFrame >= datasetTeam.historico_medias:
                idxRangeA += 1
                idxRangeB += 1

            arrDictMediasEntradas.append(dictEntrada)
            dictEntrada = dict()

            for key in ObjForDataFrame().arrKeysRotulo():
                dictRotulo[key] = objDataFrame.getDict()[key]

            arrDictMediasRotulos.append(dictRotulo)
            dictRotulo = dict()

        return arrDictMediasEntradas, arrDictMediasRotulos

    @staticmethod
    def getDictMediasDatasetsB(datasetTeam: DatasetTeam):
        dictEntrada = dict()
        dictRotulo = dict()
        arrDictMediasEntradas = []
        arrDictMediasRotulos = []
        arrObjsForDataFrame = datasetTeam.arr_obj_for_dataFrame

        for idxObjDataFrame in range(len(arrObjsForDataFrame)):
            objDataFrame = arrObjsForDataFrame[idxObjDataFrame]
            dictEntrada["is_home"] = objDataFrame.is_home
            dictEntrada["date"] = int(objDataFrame.date.timestamp() * 1000)
            dictEntrada["id_fixture"] = objDataFrame.id_fixture
            dictEntrada["id_team"] = objDataFrame.id_team
            dictEntrada["id_team_adversario"] = objDataFrame.id_team_adversario

            for key in ObjForDataFrame().arrKeysMedia():
                if idxObjDataFrame == 0:
                    dictEntrada[key] = -1
                else:
                    value = arrObjsForDataFrame[idxObjDataFrame - 1].getDict()[key]
                    dictEntrada[key] = value if value is not None else -1

            arrDictMediasEntradas.append(dictEntrada)
            dictEntrada = dict()

            for key in ObjForDataFrame().arrKeysRotulo():
                dictRotulo[key] = objDataFrame.getDict()[key]

            arrDictMediasRotulos.append(dictRotulo)
            dictRotulo = dict()

        return arrDictMediasEntradas, arrDictMediasRotulos

    @staticmethod
    def normalizarArrDatasetsTeamInBatchsA(arrDatasetTeam: list[DatasetTeam]):
        newDatasetTeam = deepcopy(arrDatasetTeam[0])
        newDatasetTeam.name_team = ""
        newDatasetTeam.dataset_entrada = []
        newDatasetTeam.dataset_rotulo = []
        newDatasetTeam.dataset_prever = []

        for datasetTeam in arrDatasetTeam:
            newDatasetTeam.dataset_entrada.append(datasetTeam.dataset_entrada)
            newDatasetTeam.dataset_rotulo.append(datasetTeam.dataset_rotulo)

            if datasetTeam.is_prever:
                for prev in datasetTeam.dataset_prever:
                    newDatasetTeam.dataset_prever.append(prev)
                newDatasetTeam.name_team += datasetTeam.name_team + " /"

        return [newDatasetTeam]

    def obterDatasetTrainB(self, arrDatasetTeam: list[DatasetTeam]):
        arrAllDictsEntradas: list[dict] = []
        arrAllDictsEntradasValues: list = []

        arrAllDictsRotulos: list[dict] = []
        arrAllDictsRotulosValues: list = []

        for datasetTeam in arrDatasetTeam:
            for dictEntrada in datasetTeam.arrDictsMediaEntrada:
                arrAllDictsEntradas.append(dictEntrada)
                arrAllDictsEntradasValues.append(list(dictEntrada.values()))

            for dictRotulo in datasetTeam.arrDictsMediaRotulo:
                arrAllDictsRotulos.append(dictRotulo)
                arrAllDictsRotulosValues.append(list(dictRotulo.values()))

        arrAllValuesNormalziados = self.normalizarEntradaTanh2D(arrOriginal=arrAllDictsEntradasValues)[0]

        arrIdsTeamsEncontrados = []
        arrEntradasValuesTeam = []
        arrRotulosValuesTeam = []

        for idxTeam, dictTeam in enumerate(arrAllDictsEntradas):
            arrDataEntradaValues = []
            arrDataRotuloValues = []
            if dictTeam["id_team"] not in arrIdsTeamsEncontrados:
                arrIdsTeamsEncontrados.append(dictTeam["id_team"])
                for idxA, dictA in enumerate(arrAllDictsEntradas):
                    if dictA["id_team"] == dictTeam["id_team"]:
                        arrDataEntradaValues.append(arrAllValuesNormalziados[idxA])
                        arrDataRotuloValues.append(arrAllDictsRotulosValues[idxA])

                arrEntradasValuesTeam.append(arrDataEntradaValues)
                arrRotulosValuesTeam.append(arrDataRotuloValues)

        newDatasetTeam = deepcopy(arrDatasetTeam[0])
        newDatasetTeam.name_team = ""

        datasetEntrada = []
        datasetRotulo = []
        datasetPrever = []
        for idxEntradasValuesTeam, entradasValuesTeam in enumerate(arrEntradasValuesTeam):
            if len(entradasValuesTeam) == 0:
                continue
            arrTimestepsEntrada = []
            arrTimestepsPrever = []
            arrTimestepsRotulo = [[] for _ in range(len(arrRotulosValuesTeam[idxEntradasValuesTeam][0]))]
            for idxValuesTeam, valuesTeam in enumerate(entradasValuesTeam):
                rangeA = idxValuesTeam
                rangeB = idxValuesTeam + arrDatasetTeam[idxEntradasValuesTeam].historico_medias

                timesteps = entradasValuesTeam[rangeA:rangeB]

                if idxValuesTeam < len(entradasValuesTeam) - arrDatasetTeam[idxEntradasValuesTeam].historico_medias:
                    arrTimestepsEntrada.append(timesteps)

                    rotuloValueTeam = arrRotulosValuesTeam[idxEntradasValuesTeam][rangeB - 1]

                    saldogols = 0
                    for idxRotuloValue, rotuloValue in enumerate(rotuloValueTeam):
                        if idxRotuloValue == 0:
                            saldogols += rotuloValue
                        elif idxRotuloValue == 1:
                            saldogols -= rotuloValue

                    rotulo = [0, 0, 0]
                    if saldogols <= -2:
                        rotulo[0] = 1
                    elif -1 <= saldogols <= 1:
                        rotulo[1] = 1
                    elif saldogols >= 2:
                        rotulo[2] = 1
                    arrTimestepsRotulo[0].append(rotulo)
                else:
                    if arrDatasetTeam[idxEntradasValuesTeam].is_prever == 1:
                        newDatasetTeam.name_team += arrDatasetTeam[idxEntradasValuesTeam].name_team + " / "
                        arrTimestepsPrever.append(timesteps)
                        break

            if len(arrTimestepsEntrada) >= 1:
                datasetEntrada.append(arrTimestepsEntrada)
                datasetRotulo.append(arrTimestepsRotulo)
            if len(arrTimestepsPrever) >= 1:
                datasetPrever.append(arrTimestepsPrever)

        newDatasetTeam.dataset_entrada = datasetEntrada
        newDatasetTeam.dataset_rotulo = datasetRotulo
        newDatasetTeam.dataset_prever = datasetPrever
        return [newDatasetTeam]

    def obterDatasetTrainC(self, arrDatasetTeam: list[DatasetTeam]):
        arrAllDictsEntradas: list[dict] = []
        arrAllDictsEntradasValues: list = []

        arrAllDictsRotulos: list[dict] = []
        arrAllDictsRotulosValues: list = []

        for datasetTeam in arrDatasetTeam:
            for dictEntrada in datasetTeam.arrDictsMediaEntrada:
                arrAllDictsEntradas.append(dictEntrada)
                arrAllDictsEntradasValues.append(list(dictEntrada.values()))

            for dictRotulo in datasetTeam.arrDictsMediaRotulo:
                arrAllDictsRotulos.append(dictRotulo)
                arrAllDictsRotulosValues.append(list(dictRotulo.values()))

        arrAllValuesNormalziados = self.normalizarEntradaTanh2D(arrOriginal=arrAllDictsEntradasValues)[0]

        arrIdsTeamsEncontrados = []
        arrEntradasValuesTeam = []
        arrRotulosValuesTeam = []

        for idxTeam, dictTeam in enumerate(arrAllDictsEntradas):
            arrDataEntradaValues = []
            arrDataRotuloValues = []
            if dictTeam["id_team"] not in arrIdsTeamsEncontrados:
                arrIdsTeamsEncontrados.append(dictTeam["id_team"])
                for idxA, dictA in enumerate(arrAllDictsEntradas):

                    if dictA["id_team"] == dictTeam["id_team"]:
                        dataEntradaTeamA = arrAllValuesNormalziados[idxA]
                        dataRotuloTeamA = arrAllDictsRotulosValues[idxA]

                        for idxB, dictB in enumerate(arrAllDictsEntradas):
                            if dictB["id_fixture"] == dictA["id_fixture"] and idxB != idxA:
                                dataEntradaTeamB = arrAllValuesNormalziados[idxB]
                                dataEntradaTeamA = numpy.concatenate((dataEntradaTeamA, dataEntradaTeamB)).tolist()

                                arrDataEntradaValues.append(dataEntradaTeamA)
                                arrDataRotuloValues.append(dataRotuloTeamA)
                arrEntradasValuesTeam.append(arrDataEntradaValues)
                arrRotulosValuesTeam.append(arrDataRotuloValues)

        newDatasetTeam = deepcopy(arrDatasetTeam[0])
        newDatasetTeam.name_team = ""

        datasetEntrada = []
        datasetRotulo = []
        datasetPrever = []
        for idxEntradasValuesTeam, entradasValuesTeam in enumerate(arrEntradasValuesTeam):
            if len(entradasValuesTeam) == 0:
                continue
            arrTimestepsEntrada = []
            arrTimestepsPrever = []
            arrTimestepsRotulo = [[]]
            for idxValuesTeam, valuesTeam in enumerate(entradasValuesTeam):
                rangeA = idxValuesTeam
                rangeB = idxValuesTeam + arrDatasetTeam[idxEntradasValuesTeam].historico_medias

                timesteps = entradasValuesTeam[rangeA:rangeB]

                if idxValuesTeam < len(entradasValuesTeam) - arrDatasetTeam[idxEntradasValuesTeam].historico_medias:
                    arrTimestepsEntrada.append(timesteps)

                    rotuloValueTeam = arrRotulosValuesTeam[idxEntradasValuesTeam][rangeB - 1]

                    saldogols = 0
                    for idxRotuloValue, rotuloValue in enumerate(rotuloValueTeam):
                        if idxRotuloValue == 0:
                            saldogols += rotuloValue
                        elif idxRotuloValue == 1:
                            saldogols -= rotuloValue

                    rotulo = [0, 0, 0]
                    if saldogols <= -2:
                        rotulo[0] = 1
                    elif -1 <= saldogols <= 1:
                        rotulo[1] = 1
                    elif saldogols >= 2:
                        rotulo[2] = 1
                    arrTimestepsRotulo[0].append(rotulo)
                else:
                    if arrDatasetTeam[idxEntradasValuesTeam].is_prever == 1:
                        newDatasetTeam.name_team += arrDatasetTeam[idxEntradasValuesTeam].name_team + " / "
                        arrTimestepsPrever.append(timesteps)
                        break

            if len(arrTimestepsEntrada) >= 1:
                datasetEntrada.append(arrTimestepsEntrada)
                datasetRotulo.append(arrTimestepsRotulo)
            if len(arrTimestepsPrever) >= 1:
                datasetPrever.append(arrTimestepsPrever)

        newDatasetTeam.dataset_entrada = datasetEntrada
        newDatasetTeam.dataset_rotulo = datasetRotulo
        newDatasetTeam.dataset_prever = datasetPrever
        return [newDatasetTeam]

    def obterDatasetTrainD(self, arrDatasetTeam: list[DatasetTeam]):
        arrAllDictsEntradas: list[dict] = []
        arrAllDictsEntradasValues: list = []

        arrAllDictsRotulos: list[dict] = []
        arrAllDictsRotulosValues: list = []

        for datasetTeam in arrDatasetTeam:
            for dictEntrada in datasetTeam.arrDictsMediaEntrada:
                arrAllDictsEntradas.append(dictEntrada)
                arrAllDictsEntradasValues.append(list(dictEntrada.values()))

            for dictRotulo in datasetTeam.arrDictsMediaRotulo:
                arrAllDictsRotulos.append(dictRotulo)
                arrAllDictsRotulosValues.append(list(dictRotulo.values()))

        arrAllValuesNormalziados = self.normalizarEntradaTanh2D(arrOriginal=arrAllDictsEntradasValues)[0]

        arrIdsTeamsEncontrados = []
        arrEntradasValuesTeam = []
        arrRotulosValuesTeam = []

        for idxTeam, dictTeam in enumerate(arrAllDictsEntradas):
            arrDataEntradaValues = []
            arrDataRotuloValues = []
            if dictTeam["id_team"] not in arrIdsTeamsEncontrados:
                arrIdsTeamsEncontrados.append(dictTeam["id_team"])
                for idxA, dictA in enumerate(arrAllDictsEntradas):

                    if dictA["id_team"] == dictTeam["id_team"]:
                        timestepEntrada = [arrAllValuesNormalziados[idxA]]
                        dataRotuloTeamA = arrAllDictsRotulosValues[idxA]

                        for idxB, dictB in enumerate(arrAllDictsEntradas):
                            if dictB["id_fixture"] == dictA["id_fixture"] and idxB != idxA:
                                timestepEntrada.append(arrAllValuesNormalziados[idxB])

                                arrDataEntradaValues.append(timestepEntrada)
                                arrDataRotuloValues.append(dataRotuloTeamA)
                arrEntradasValuesTeam.append(arrDataEntradaValues)
                arrRotulosValuesTeam.append(arrDataRotuloValues)

        newDatasetTeam = deepcopy(arrDatasetTeam[0])
        newDatasetTeam.name_team = ""

        datasetEntrada = []
        datasetRotulo = []
        datasetPrever = []
        for idxEntradasValuesTeam, entradasValuesTeam in enumerate(arrEntradasValuesTeam):
            if len(entradasValuesTeam) == 0 or arrDatasetTeam[idxEntradasValuesTeam].is_prever == 0:
                continue

            arrTimestepsEntrada = []
            arrTimestepsPrever = []
            arrTimestepsRotulo = [[]]
            for idxValuesTeam, valuesTeam in enumerate(entradasValuesTeam):

                if idxValuesTeam < len(entradasValuesTeam) - 1:
                    arrTimestepsEntrada.append(valuesTeam)
                    rotuloValueTeam = arrRotulosValuesTeam[idxEntradasValuesTeam][idxValuesTeam]

                    saldogols = 0
                    for idxRotuloValue, rotuloValue in enumerate(rotuloValueTeam):
                        if idxRotuloValue == 0:
                            saldogols += rotuloValue
                        elif idxRotuloValue == 1:
                            saldogols -= rotuloValue

                    rotulo = [0, 0, 0, 0, 0]
                    if saldogols <= -3:
                        rotulo[0] = 1
                    elif saldogols <= -1:
                        rotulo[1] = 1
                    elif saldogols == 0:
                        rotulo[2] = 1
                    elif saldogols >= 3:
                        rotulo[4] = 1
                    elif saldogols >= 1:
                        rotulo[3] = 1
                    arrTimestepsRotulo[0].append(rotulo)
                else:
                    if arrDatasetTeam[idxEntradasValuesTeam].is_prever == 1:
                        newDatasetTeam.name_team += arrDatasetTeam[idxEntradasValuesTeam].name_team + " / "
                        arrTimestepsPrever.append(valuesTeam)
                        break

            if len(arrTimestepsEntrada) >= 1:
                datasetEntrada.append(arrTimestepsEntrada)
                datasetRotulo.append(arrTimestepsRotulo)
            if len(arrTimestepsPrever) >= 1:
                datasetPrever.append(arrTimestepsPrever)

        newDatasetTeam.dataset_entrada = datasetEntrada
        newDatasetTeam.dataset_rotulo = datasetRotulo
        newDatasetTeam.dataset_prever = datasetPrever
        return [newDatasetTeam]

    def obterDatasetTrainE(self, arrDatasetTeam: list[DatasetTeam]):
        arrAllDictsEntradas: list[dict] = []
        arrAllDictsEntradasValues: list = []

        arrAllDictsRotulos: list[dict] = []
        arrAllDictsRotulosValues: list = []

        for datasetTeam in arrDatasetTeam:
            for dictEntrada in datasetTeam.arrDictsMediaEntrada:
                arrAllDictsEntradas.append(dictEntrada)
                arrAllDictsEntradasValues.append(list(dictEntrada.values()))

            for dictRotulo in datasetTeam.arrDictsMediaRotulo:
                arrAllDictsRotulos.append(dictRotulo)
                arrAllDictsRotulosValues.append(list(dictRotulo.values()))

        arrAllValuesNormalziados = self.normalizarEntradaTanh2D(arrOriginal=arrAllDictsEntradasValues)[0]

        arrIdsTeamsEncontrados = []
        arrEntradasValuesTeam = []
        arrRotulosValuesTeam = []

        for idxTeam, dictTeam in enumerate(arrAllDictsEntradas):
            arrDataEntradaValues = []
            arrDataRotuloValues = []
            if dictTeam["id_team"] not in arrIdsTeamsEncontrados:
                arrIdsTeamsEncontrados.append(dictTeam["id_team"])
                for idxA, dictA in enumerate(arrAllDictsEntradas):
                    if dictA["id_team"] == dictTeam["id_team"]:
                        arrDataEntradaValues.append(arrAllValuesNormalziados[idxA])
                        arrDataRotuloValues.append(arrAllDictsRotulosValues[idxA])

                arrEntradasValuesTeam.append(arrDataEntradaValues)
                arrRotulosValuesTeam.append(arrDataRotuloValues)

        newDatasetTeam = deepcopy(arrDatasetTeam[0])
        newDatasetTeam.name_team = ""

        datasetEntrada = []
        datasetRotulo = []
        datasetPrever = []
        for idxEntradasValuesTeam, entradasValuesTeam in enumerate(arrEntradasValuesTeam):
            arrTimestepsEntrada = []
            arrTimestepsPrever = []
            arrTimestepsRotulo = [[]]
            for idxValuesTeam, valuesTeam in enumerate(entradasValuesTeam):
                rangeA = idxValuesTeam
                rangeB = idxValuesTeam + arrDatasetTeam[idxEntradasValuesTeam].historico_medias

                timesteps = entradasValuesTeam[rangeA:rangeB]

                if idxValuesTeam < len(entradasValuesTeam) - arrDatasetTeam[idxEntradasValuesTeam].historico_medias:
                    arrTimestepsEntrada.append(timesteps)

                    rotuloValueTeam = arrRotulosValuesTeam[idxEntradasValuesTeam][rangeB - 1]

                    saldogols = 0
                    for idxRotuloValue, rotuloValue in enumerate(rotuloValueTeam):
                        if idxRotuloValue == 0:
                            saldogols += rotuloValue
                        elif idxRotuloValue == 1:
                            saldogols -= rotuloValue

                    rotulo = [0, 0, 0, 0, 0]
                    if saldogols <= -3:
                        rotulo[0] = 1
                    elif saldogols <= -1:
                        rotulo[1] = 1
                    elif saldogols == 0:
                        rotulo[2] = 1
                    elif saldogols >= 3:
                        rotulo[4] = 1
                    elif saldogols >= 1:
                        rotulo[3] = 1
                    arrTimestepsRotulo[0].append(rotulo)
                else:
                    if arrDatasetTeam[idxEntradasValuesTeam].is_prever == 1:
                        newDatasetTeam.name_team += arrDatasetTeam[idxEntradasValuesTeam].name_team + " / "
                        arrTimestepsPrever.append(timesteps)
                        break

            if len(arrTimestepsEntrada) >= 1:
                datasetEntrada.append(arrTimestepsEntrada)
                datasetRotulo.append(arrTimestepsRotulo)
            if len(arrTimestepsPrever) >= 1:
                datasetPrever.append(arrTimestepsPrever)

        newDatasetTeam.dataset_entrada = datasetEntrada
        newDatasetTeam.dataset_rotulo = datasetRotulo
        newDatasetTeam.dataset_prever = datasetPrever
        return [newDatasetTeam]

    # Cópia do C mas com filtro id_fixture
    def obterDatasetTrainF(self, arrDatasetTeam: list[DatasetTeam]):
        arrAllDictsEntradas: list[dict] = []
        arrAllDictsEntradasValues: list = []

        arrAllDictsRotulos: list[dict] = []
        arrAllDictsRotulosValues: list = []

        for datasetTeam in arrDatasetTeam:
            for dictEntrada in datasetTeam.arrDictsMediaEntrada:
                arrAllDictsEntradas.append(dictEntrada)
                arrAllDictsEntradasValues.append(list(dictEntrada.values()))

            for dictRotulo in datasetTeam.arrDictsMediaRotulo:
                arrAllDictsRotulos.append(dictRotulo)
                arrAllDictsRotulosValues.append(list(dictRotulo.values()))

        arrAllValuesNormalziados = self.normalizarEntradaTanh2D(arrOriginal=arrAllDictsEntradasValues)[0]

        arrIdsTeamsEncontrados = []
        arrIdsFixturesEncontrados = []
        arrEntradasValuesTeam = []
        arrRotulosValuesTeam = []

        for idxTeam, dictTeam in enumerate(arrAllDictsEntradas):
            arrDataEntradaValues = []
            arrDataRotuloValues = []
            if dictTeam["id_team"] not in arrIdsTeamsEncontrados:
                arrIdsTeamsEncontrados.append(dictTeam["id_team"])
                for idxA, dictA in enumerate(arrAllDictsEntradas):
                    '''if dictA["id_fixture"] in arrIdsFixturesEncontrados:
                        continue'''

                    if dictA["id_team"] == dictTeam["id_team"]:
                        dataEntradaTeamA = arrAllValuesNormalziados[idxA]
                        dataRotuloTeamA = arrAllDictsRotulosValues[idxA]

                        '''for idxB, dictB in enumerate(arrAllDictsEntradas):
                            if dictB["id_fixture"] == dictA["id_fixture"] and idxB != idxA:
                                arrIdsFixturesEncontrados.append(dictA["id_fixture"])
                                dataEntradaTeamB = arrAllValuesNormalziados[idxB]
                                dataEntradaTeamA = numpy.concatenate((dataEntradaTeamA, dataEntradaTeamB)).tolist()'''

                        arrDataEntradaValues.append(dataEntradaTeamA)
                        arrDataRotuloValues.append(dataRotuloTeamA)
                arrEntradasValuesTeam.append(arrDataEntradaValues)
                arrRotulosValuesTeam.append(arrDataRotuloValues)

        newDatasetTeam = deepcopy(arrDatasetTeam[0])
        newDatasetTeam.name_team = ""

        datasetEntrada = []
        datasetRotulo = []
        datasetPrever = []
        for idxEntradasValuesTeam, entradasValuesTeam in enumerate(arrEntradasValuesTeam):
            if len(entradasValuesTeam) == 0:
                continue
            arrTimestepsEntrada = []
            arrTimestepsPrever = []
            arrTimestepsRotulo = [[]]
            for idxValuesTeam, valuesTeam in enumerate(entradasValuesTeam):
                rangeA = idxValuesTeam
                rangeB = idxValuesTeam + arrDatasetTeam[idxEntradasValuesTeam].historico_medias

                timesteps = entradasValuesTeam[rangeA:rangeB]

                if idxValuesTeam < len(entradasValuesTeam) - arrDatasetTeam[idxEntradasValuesTeam].historico_medias:
                    arrTimestepsEntrada.append(timesteps)

                    rotuloValueTeam = arrRotulosValuesTeam[idxEntradasValuesTeam][rangeB - 1]

                    saldogols = 0
                    for idxRotuloValue, rotuloValue in enumerate(rotuloValueTeam):
                        if idxRotuloValue == 0:
                            saldogols += rotuloValue
                        elif idxRotuloValue == 1:
                            saldogols -= rotuloValue

                    rotulo = [0, 0, 0, 0, 0]
                    if saldogols <= -3:
                        rotulo[0] = 1
                    elif saldogols <= -1:
                        rotulo[1] = 1
                    elif saldogols == 0:
                        rotulo[2] = 1
                    elif saldogols <= 2:
                        rotulo[3] = 1
                    elif saldogols >= 3:
                        rotulo[4] = 1
                    arrTimestepsRotulo[0].append(rotulo)
                else:
                    if arrDatasetTeam[idxEntradasValuesTeam].is_prever == 1:
                        newDatasetTeam.name_team += arrDatasetTeam[idxEntradasValuesTeam].name_team + " / "
                        arrTimestepsPrever.append(timesteps)
                        break

            if len(arrTimestepsEntrada) >= 1:
                datasetEntrada.append(arrTimestepsEntrada)
                datasetRotulo.append(arrTimestepsRotulo)
            if len(arrTimestepsPrever) >= 1:
                datasetPrever.append(arrTimestepsPrever)

        newDatasetTeam.dataset_entrada = datasetEntrada
        newDatasetTeam.dataset_rotulo = datasetRotulo
        newDatasetTeam.dataset_prever = datasetPrever
        return [newDatasetTeam]

    def obterDatasetTrainG(self, arrDatasetTeam: list[DatasetTeam]):
        for datasetTeamData in arrDatasetTeam:
            arrAllDictsEntradas: list[dict] = []
            arrAllDictsEntradasValues: list = []

            arrAllDictsRotulos: list[dict] = []
            arrAllDictsRotulosValues: list = []

            for datasetTeam in arrDatasetTeam:
                for dictEntrada in datasetTeam.arrDictsMediaEntrada:
                    arrAllDictsEntradas.append(dictEntrada)
                    arrAllDictsEntradasValues.append(list(dictEntrada.values()))

                for dictRotulo in datasetTeam.arrDictsMediaRotulo:
                    arrAllDictsRotulos.append(dictRotulo)
                    arrAllDictsRotulosValues.append(list(dictRotulo.values()))

            arrAllValuesNormalziados = self.normalizarEntradaTanh2D(arrOriginal=arrAllDictsEntradasValues)[0]

            arrEntradasValuesTeam = []
            arrRotulosValuesTeam = []
            arrPreverValuesTeam = []

            for idxTeam, dictTeam in enumerate(arrAllDictsEntradas):
                arrTimesteps = []
                if dictTeam["id_team"] == datasetTeamData.id_team:
                    for idxTeamB, dictTeamB in enumerate(arrAllDictsEntradas):

                        if idxTeam != idxTeamB and dictTeam["id_fixture"] == dictTeamB["id_fixture"]:
                            dataEntradaTeamA = arrAllValuesNormalziados[idxTeam]
                            dataEntradaTeamB = arrAllValuesNormalziados[idxTeamB]
                            dataRotuloTeamA = arrAllDictsRotulosValues[idxTeam]

                            arrTimesteps.append(dataEntradaTeamA)
                            arrTimesteps.append(dataEntradaTeamB)

                            if dictTeam["id_fixture"] != datasetTeamData.arr_obj_for_dataFrame[-1].id_fixture:
                                arrEntradasValuesTeam.append([arrTimesteps])
                                arrRotulosValuesTeam.append(dataRotuloTeamA)
                            else:
                                arrPreverValuesTeam.append(arrTimesteps)

            datasetEntrada = arrEntradasValuesTeam
            datasetRotulo = []
            datasetPrever = [arrPreverValuesTeam]
            for idxRotulosValuesTeam, rotulosValuesTeam in enumerate(arrRotulosValuesTeam):
                arrTimestepsRotulo = [[], [], []]
                saldogols = 0
                sumGoals = 0
                sumGoalsMarcados = 0
                sumGoalsSofridos = 0
                for idxValuesTeam, valueRotulo in enumerate(rotulosValuesTeam):
                    sumGoals += valueRotulo
                    if idxValuesTeam == 0:
                        saldogols += valueRotulo
                        sumGoalsMarcados += valueRotulo
                    elif idxValuesTeam == 1:
                        saldogols -= valueRotulo
                        sumGoalsSofridos += valueRotulo

                '''rotuloA = [0, 0, 0, 0, 0]
                if saldogols <= -3:
                    rotuloA[0] = 1
                elif saldogols <= -1:
                    rotuloA[1] = 1
                elif saldogols == 0:
                    rotuloA[2] = 1
                elif saldogols <= 2:
                    rotuloA[3] = 1
                elif saldogols >= 3:
                    rotuloA[4] = 1'''

                rotuloA = [0, 0, 0, 0, 0]
                if sumGoals <= 3:
                    rotuloA[sumGoals] = 1
                else:
                    rotuloA[4] = 1

                rotuloB = [0, 0, 0, 0, 0]
                if sumGoalsMarcados <= 3:
                    rotuloB[sumGoalsMarcados] = 1
                else:
                    rotuloB[4] = 1

                rotuloC = [0, 0, 0, 0, 0]
                if sumGoalsSofridos <= 3:
                    rotuloC[sumGoalsSofridos] = 1
                else:
                    rotuloC[4] = 1

                '''rotuloB = [0, 0]
                if saldogols <= -1:
                    rotuloB[0] = 1
                elif saldogols >= 0:
                    rotuloB[1] = 1'''

                arrTimestepsRotulo[0].append(rotuloA)
                arrTimestepsRotulo[1].append(rotuloB)
                arrTimestepsRotulo[2].append(rotuloC)
                # arrTimestepsRotulo[1].append(rotuloB)
                datasetRotulo.append(arrTimestepsRotulo)

            datasetTeamData.dataset_entrada = datasetEntrada
            datasetTeamData.dataset_rotulo = datasetRotulo
            datasetTeamData.dataset_prever = datasetPrever
        return arrDatasetTeam

    @staticmethod
    def normalizarEntradaTanh3D(arrOriginal: list, valores_minimos=None, amplitudes=None):
        if valores_minimos is None:
            valores_minimos = numpy.min(arrOriginal, axis=(0, 1))

        if amplitudes is None:
            amplitudes = numpy.max(arrOriginal, axis=(0, 1)) - numpy.array(valores_minimos)

        matriz_normalizada = -1 + 2 * (arrOriginal - numpy.array(valores_minimos)) / numpy.array(amplitudes)
        return matriz_normalizada.tolist(), valores_minimos, amplitudes

    @staticmethod
    def normalizarEntradaTanh2D(arrOriginal: list | numpy.ndarray, valores_minimos=None, amplitudes=None):
        if valores_minimos is None:
            valores_minimos = numpy.min(arrOriginal, axis=0)

        if amplitudes is None:
            amplitudes = numpy.max(numpy.array(arrOriginal), axis=0) - numpy.array(valores_minimos)
            amplitudes = numpy.maximum(amplitudes, 1e-5)

        matriz_normalizada = -1 + 2 * (arrOriginal - numpy.array(valores_minimos)) / numpy.array(amplitudes)
        matriz_normalizada_cleaned = numpy.where(numpy.isinf(matriz_normalizada) | numpy.isnan(matriz_normalizada), 0,
                                                 numpy.where(numpy.isnan(matriz_normalizada), 0, matriz_normalizada))

        return matriz_normalizada_cleaned.tolist(), valores_minimos, amplitudes

    @staticmethod
    def normalizarEntradaSigmoid3D(arrOriginal: list, valores_minimos=None, amplitudes=None):
        if valores_minimos is None:
            valores_minimos = numpy.min(numpy.array(arrOriginal), axis=(0, 1))

        if amplitudes is None:
            amplitudes = numpy.max(numpy.array(arrOriginal), axis=(0, 1)) - numpy.array(valores_minimos)
        matriz_normalizada = (numpy.array(arrOriginal) - numpy.array(valores_minimos)) / numpy.array(amplitudes)
        matriz_normalizada_cleaned = numpy.where(numpy.isinf(matriz_normalizada) | numpy.isnan(matriz_normalizada), 0,
                                                 numpy.where(numpy.isnan(matriz_normalizada), 0, matriz_normalizada))

        return matriz_normalizada_cleaned.tolist(), valores_minimos, amplitudes

    @staticmethod
    def normalizarEntradaSigmoid2D(arrOriginal: list | numpy.ndarray, valores_minimos=None, amplitudes=None):
        if valores_minimos is None:
            valores_minimos = numpy.min(numpy.array(arrOriginal), axis=0)

        if amplitudes is None:
            amplitudes = numpy.max(numpy.array(arrOriginal), axis=0) - numpy.array(valores_minimos)
            amplitudes = numpy.maximum(amplitudes, 1e-5)
        matriz_normalizada = (numpy.array(arrOriginal) - numpy.array(valores_minimos)) / numpy.array(amplitudes)
        matriz_normalizada_cleaned = numpy.where(numpy.isinf(matriz_normalizada) | numpy.isnan(matriz_normalizada), 0,
                                                 numpy.where(numpy.isnan(matriz_normalizada), 0, matriz_normalizada))

        return matriz_normalizada_cleaned.tolist(), valores_minimos, amplitudes
