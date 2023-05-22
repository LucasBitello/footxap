from __future__ import annotations

from copy import  deepcopy

from api.models.fixturesModel import Fixture
from api.models.fixturesTeamsModel import FixtureTeams
from api.models.fixturesTeamsStatisticsModel import FixtureTeamStatistic
from api.models.teamsModel import Team
from api.models.typeFixtureTeamStatisticModel import TypesFixturesTeamsStatisticsModel, TypesFixturesTeamsStatistics

from api.regras.fixturesRegras import FixturesRegras
from api.regras.teamsRegras import TeamsRegras

class DatasetDataFixtureTeamSaida:
    def __init__(self):
        self.is_winner: int = None
class DatasetDataFixtureTeam:
    def __init__(self):
        self.id_fixture: int = None
        self.data_fixture: str = None
        self.is_jogando_home: int = None

        self.id_team: int = None
        self.pontos: int = None
        self.saldo_gols: int = None
        self.qtde_gols_marcados: int = None
        self.qtde_gols_sofridos: int = None
        self.qtde_vitoria: int = None
        self.qtde_empates: int = None
        self.qtde_derrotas: int = None

        self.arr_estatisticas: list[FixtureTeamStatistic] = []
        self.media_estatisticas: list[dict] = []

        self.datasetDataFixtureSaida: DatasetDataFixtureTeamSaida = DatasetDataFixtureTeamSaida()

class DatasetDataFixtureByTeam:
    def __init__(self):
        self.id_team: int = None
        self.name_team: int = None
        self.arr_dataset_data_fixture_team: list[DatasetDataFixtureTeam] = []

class DatasetFixtureRegras:
    def __init__(self):
        self.fixturesRegras = FixturesRegras()
        self.teamsRegras = TeamsRegras()


    def criarDatasetFixture(self, arr_ids_team: list = [], qtde_dados_team: int = 15,
                            isAtualizarFixturesOutherTeam: bool = True, isFiltrarApenasTeams: bool = False):
        arrDatasetFixtureTeam: list[DatasetDataFixtureTeam] = []

        for id_team in arr_ids_team:
            self.fixturesRegras.fixturesModel.atualizarDados(arr_ids_team=[id_team],
                                                             qtde_dados_estatisticas=qtde_dados_team)

            arrFixtures: list[Fixture] = self.fixturesRegras.obter(id_team=id_team, isAsc=False, limit=qtde_dados_team,
                                                                   isApenasComStatistics=True, isObterProximoJogo=True)

            for fixture in list(reversed(arrFixtures)):
                fixture.teams: list[FixtureTeams] = fixture.teams

                for teamFixture in fixture.teams:
                    if teamFixture.id_team != id_team and isAtualizarFixturesOutherTeam:
                        self.fixturesRegras.fixturesModel.atualizarDados(arr_ids_team=[teamFixture.id_team],
                                                                         qtde_dados_estatisticas=qtde_dados_team)

                    if teamFixture.id_team == id_team:
                        newDatasetFixtureTeam = DatasetDataFixtureTeam()
                        newDatasetFixtureTeam.data_fixture = fixture.date.strftime("%Y-%m-%d")
                        newDatasetFixtureTeam.id_team = id_team
                        newDatasetFixtureTeam.id_fixture = fixture.id
                        newDatasetFixtureTeam.is_jogando_home = teamFixture.is_home
                        newDatasetFixtureTeam.datasetDataFixtureSaida.is_winner = \
                            self.normalizarIsWinner(isWinnerDB=teamFixture.is_winner)
                        newDatasetFixtureTeam.arr_estatisticas = \
                            self.fixturesRegras.obterFixtureEstatisticasByIdFixtureIdTeam(id_fixture=fixture.id,
                                                                                          id_team=id_team)

                        datasetJaAdicionado = self.filtrarDataFixtureTeam(arrDatasetDataFixture=arrDatasetFixtureTeam,
                                                                          id_fixture=fixture.id, id_team=id_team)

                        if datasetJaAdicionado is None and (len(newDatasetFixtureTeam.arr_estatisticas) >= 1 or fixture.status == "NS"):
                            arrDatasetFixtureTeam.append(newDatasetFixtureTeam)

                    else:
                        arrFixturesOutherTeam: list[Fixture] = self.fixturesRegras.obter(id_team=teamFixture.id_team,
                                                                                         isAsc=False,
                                                                                         limit=qtde_dados_team,
                                                                                         isApenasComStatistics=True,
                                                                                         isObterProximoJogo=True)

                        for fixtureOuterTeam in list(reversed(arrFixturesOutherTeam)):
                            fixtureOuterTeam.teams: list[FixtureTeams] = fixtureOuterTeam.teams

                            for outherTeamFixture in fixtureOuterTeam.teams:
                                if outherTeamFixture.id_team == teamFixture.id_team:
                                    newDatasetFixtureTeam = DatasetDataFixtureTeam()
                                    newDatasetFixtureTeam.data_fixture = fixtureOuterTeam.date.strftime("%Y-%m-%d")
                                    newDatasetFixtureTeam.id_team = outherTeamFixture.id_team
                                    newDatasetFixtureTeam.id_fixture = fixtureOuterTeam.id
                                    newDatasetFixtureTeam.is_jogando_home = outherTeamFixture.is_home
                                    newDatasetFixtureTeam.datasetDataFixtureSaida.is_winner = \
                                        self.normalizarIsWinner(isWinnerDB=outherTeamFixture.is_winner)
                                    newDatasetFixtureTeam.arr_estatisticas = \
                                        self.fixturesRegras.obterFixtureEstatisticasByIdFixtureIdTeam(id_fixture=fixtureOuterTeam.id,
                                                                                                      id_team=outherTeamFixture.id_team)

                                    datasetJaAdicionado = self.filtrarDataFixtureTeam(
                                        arrDatasetDataFixture=arrDatasetFixtureTeam,
                                        id_fixture=fixtureOuterTeam.id,
                                        id_team=outherTeamFixture.id_team)

                                    if datasetJaAdicionado is None and (len(newDatasetFixtureTeam.arr_estatisticas) >= 1 or fixtureOuterTeam.status == "NS"):
                                        arrDatasetFixtureTeam.append(newDatasetFixtureTeam)

        arrDatasetDataFixtureByTeam: list[DatasetDataFixtureByTeam] = []

        for datasetDataFixtureTeam in arrDatasetFixtureTeam:
            isEncontrou = False

            for datasetDataFixtureByTeam in arrDatasetDataFixtureByTeam:
                if datasetDataFixtureByTeam.id_team == datasetDataFixtureTeam.id_team:
                    datasetDataFixtureByTeam.arr_dataset_data_fixture_team.append(datasetDataFixtureTeam)
                    isEncontrou = True
                    break

            if not isEncontrou:
                newDatasetDataFixtureByTeam = DatasetDataFixtureByTeam()
                newDatasetDataFixtureByTeam.id_team = datasetDataFixtureTeam.id_team
                newDatasetDataFixtureByTeam.name_team = self.teamsRegras.obter(id=datasetDataFixtureTeam.id_team)[0].name
                newDatasetDataFixtureByTeam.arr_dataset_data_fixture_team.append(datasetDataFixtureTeam)
                arrDatasetDataFixtureByTeam.append(newDatasetDataFixtureByTeam)

        arrFiltradoTeams = []
        for datasetDataFixtureByTeam in arrDatasetDataFixtureByTeam:
            datasetDataFixtureByTeam = self.calcularMediaStatisticsTeam(datasetDataFixtureByTeam=datasetDataFixtureByTeam)

            if datasetDataFixtureByTeam.id_team in arr_ids_team:
                arrFiltradoTeams.append(datasetDataFixtureByTeam)

        if isFiltrarApenasTeams:
            return arrFiltradoTeams
        else:
            return arrDatasetDataFixtureByTeam

    def normalizarIsWinner(self, isWinnerDB: int):
        if isWinnerDB is None:
            return 1
        elif isWinnerDB == 1:
            return 2
        elif isWinnerDB == 0:
            return 0
        else:
            print("impossivel normalizar isWiiner: ", isWinnerDB)
            raise "impossivel normalizar isWiiner"

    def filtrarDataFixtureTeam(self, arrDatasetDataFixture: list[DatasetDataFixtureTeam],
                                      id_fixture: int, id_team: int) -> DatasetDataFixtureTeam:

        datasetEncontrado: DatasetDataFixtureTeam = None

        for dataset in arrDatasetDataFixture:
            if dataset.id_fixture == id_fixture and dataset.id_team == id_team:
                datasetEncontrado = dataset

        return datasetEncontrado


    def calcularMediaStatisticsTeam(self, datasetDataFixtureByTeam: DatasetDataFixtureByTeam, qtde_jogos: int = 5) -> DatasetDataFixtureByTeam:
        arrTypeStatistics: list[TypesFixturesTeamsStatistics] = self.fixturesRegras.fixturesModel.fixturesTeamsStatisticsModel.typesFixturesTeamsStatisticsModel.obterTudo()

        inicioFatia: int = 0
        fimFatia: int = 0

        for dataFixture in datasetDataFixtureByTeam.arr_dataset_data_fixture_team:
            if fimFatia == 0:
                fimFatia += 1
                continue

            arrFatiaDataFixture = datasetDataFixtureByTeam.arr_dataset_data_fixture_team[inicioFatia:fimFatia]

            fimFatia += 1
            if fimFatia > 5:
                inicioFatia += 1


            for typeStatistics in arrTypeStatistics:
                idConcatenar = str(typeStatistics.id)
                dicMediaStatistics = {}
                dicMediaStatistics["id_statistic"] = typeStatistics.id
                dicMediaStatistics["name_statistic"] = typeStatistics.name_statistic
                dicMediaStatistics["soma_"+idConcatenar] = 0
                dicMediaStatistics["media_" + idConcatenar] = 0
                dicMediaStatistics["qtde_dados"] = len(arrFatiaDataFixture)
                dicMediaStatistics["inclinacao_media"] = None
                dicMediaStatistics["is_caindo"] = None
                dicMediaStatistics["is_decline_good"] = typeStatistics.is_decline_good == 1
                dicMediaStatistics["is_porcentagem"] = False

                for dataFatia in arrFatiaDataFixture:
                    for statistic in dataFatia.arr_estatisticas:
                        if statistic.id_type_statistic == typeStatistics.id:
                            if statistic.value is None:
                                dicMediaStatistics["soma_"+idConcatenar] += 0
                            else:
                                dicMediaStatistics["soma_"+idConcatenar] += statistic.value
                                if statistic.value > 0 and statistic.value < 1.0:
                                    dicMediaStatistics["is_porcentagem"] = True

                dicMediaStatistics["media_" + idConcatenar] = dicMediaStatistics["soma_"+idConcatenar] / dicMediaStatistics["qtde_dados"]

                if dicMediaStatistics["is_porcentagem"]:
                    dicMediaStatistics["media_" + idConcatenar + "_formatada"] = f"{dicMediaStatistics['media_' + idConcatenar] * 100:.1f}%"
                else:
                    dicMediaStatistics["media_" + idConcatenar + "_formatada"] = f"{dicMediaStatistics['media_' + idConcatenar]:.1f}"

                for statisticAnterior in arrFatiaDataFixture[len(arrFatiaDataFixture) - 1].media_estatisticas:
                    if statisticAnterior["id_statistic"] == typeStatistics.id:
                        dicMediaStatistics["inclinacao_media"], dicMediaStatistics["is_caindo"] = self.calcular_inclinacao_media(
                            valor_atual=dicMediaStatistics["media_" + idConcatenar],
                            valor_anterior=statisticAnterior["media_" + idConcatenar])

                dataFixture.media_estatisticas.append(dicMediaStatistics)

        return datasetDataFixtureByTeam

    def calcular_inclinacao_media(self, valor_atual: int | float | None, valor_anterior: int | float | None) -> list[float, bool]:

        if valor_anterior == 0:
            return 1.0, False
        elif valor_atual is None and valor_anterior is None:
            return 0, False

        diferenca_percentual = ((valor_atual - valor_anterior) / valor_anterior)
        isCaindo: bool = diferenca_percentual < 0
        diferenca_percentual = -diferenca_percentual if isCaindo else diferenca_percentual
        return diferenca_percentual, isCaindo