from __future__ import annotations
from datetime import datetime, timedelta

from api.models.model import Model, ReferenciaDatabaseToAPI, ReferenciaTabelasFilhas, IdTabelas, ReferenciaTabelasPai, ClassModel
from api.models.leaguesModel import LeaguesModel, League
from api.models.seasonsModel import SeasonsModel, Season
from api.models.fixturesTeamsModel import FixturesTeamsModel, FixtureTeams
from api.models.fixturesTeamsLineupsModel import FixturesTeamsLineupsModel, FixtureTeamLineup
from api.models.fixturesTeamsStatisticsModel import FixturesTeamsStatisticsModel, FixtureTeamStatistic
from api.models.teamsSeasonsModel import TeamsSeasonsModel, TeamSeason
from api.models.teamsModel import TeamsModel, Team
from api.models.countriesModel import Country, CountriesModel

class FixturesModel(Model):
    def __init__(self, teamsModel: object):
        self.countriesModel = CountriesModel()
        self.leaguesModel = LeaguesModel()
        self.seasonsModel = SeasonsModel()
        self.teamsModel = TeamsModel()
        self.teamsSeasonsModel = TeamsSeasonsModel()

        super().__init__(
            name_table="fixture",
            id_tabela=IdTabelas().fixture,
            name_columns_id=["id"],
            reference_db_api=[ReferenciaDatabaseToAPI(nome_coluna_db="id_api",
                                                      nome_coluna_api="id")],

            referencia_tabelas_pai=[ReferenciaTabelasPai(IdTabelas().season,
                                                         nome_tabela_pai="season",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_season")],

            referencia_tabelas_filhas=[ReferenciaTabelasFilhas(id_tabela_filha=IdTabelas().fixture_teams,
                                                               nome_tabela_filha="fixture_teams",
                                                               nome_coluna_tabela_filha="id_fixture",
                                                               nome_coluna_tabela_pai="id"),
                                       ReferenciaTabelasFilhas(id_tabela_filha=IdTabelas().fixture_team_lineups,
                                                               nome_tabela_filha="fixture_team_lineups",
                                                               nome_coluna_tabela_filha="id_fixture",
                                                               nome_coluna_tabela_pai="id"),
                                       ReferenciaTabelasFilhas(id_tabela_filha=IdTabelas().fixture_team_estatistics,
                                                               nome_tabela_filha="fixture_team_statistics",
                                                               nome_coluna_tabela_filha="id_fixture",
                                                               nome_coluna_tabela_pai="id")],
            classModelDB=Fixture,
            rate_refesh_table_in_ms=0)

        self.criarTableDataBase()
        self.fixturesTeamsModel = FixturesTeamsModel(teamsModel=teamsModel)
        self.fixturesTeamsStatisticsModel = FixturesTeamsStatisticsModel(fixtureModel=self, teamModel=teamsModel)
        '''self.fixturesTeamsLineupsModel = FixturesTeamsLineupsModel()'''


    def fazerConsultaFixturesApiFootball(self, id_fixture: int = None, date: str = None, id_league: int = None,
                                         year_season: int = None, id_team: int = None, round: str = None,
                                         timezone: str = None, last: int = None):
        """
            Date deve ser no formato YYYY-MM-DD
            Round consultar https://www.api-football.com/documentation-v3#tag/Fixtures/operation/get-fixtures
        """
        arrParams = []
        query = "fixtures"
        nameColumnResponseData = "response"

        if id_fixture is not None:
            arrParams.append("id=" + str(id_fixture))
        if date is not None:
            arrParams.append("date=" + date)
        if id_league is not None:
            arrParams.append("league=" + str(id_league))
        if year_season is not None:
            arrParams.append("season=" + str(year_season))
        if id_team is not None:
            arrParams.append("team=" + str(id_team))
        if round is not None:
            arrParams.append("round=" + round)
        if timezone is not None:
            arrParams.append("timezone=" + timezone)
        if last is not None:
            arrParams.append("last=" + last)

        if len(arrParams) >= 1:
            query += "?" + "&".join(arrParams)

        response = self.regraApiFootBall.conecarAPIFootball(query)
        responseData = response[nameColumnResponseData]

        return responseData


    def atualizarDBFixtures(self, idSeason: int):
        season: Season = self.seasonsModel.obterByColumnsID([idSeason])[0]
        league: League = self.leaguesModel.obterByColumnsID([season.id_league])[0]

        if league.id_api in [10, 666, 667]:
            return

        arrFixtures = self.fazerConsultaFixturesApiFootball(id_league=league.id_api, year_season=season.year)

        for fixture in arrFixtures:
            dataFixtureFormatada = datetime.fromisoformat(fixture["fixture"]["date"]).strftime(
                "%Y-%m-%d %H:%M:%S")

            newFixture = Fixture()
            arrFixtureDB = self.obterByReferenceApi(dadosBusca=[fixture["fixture"]["id"]])

            if len(arrFixtureDB) == 1:
                newFixture = arrFixtureDB[0]

            newFixture.id_api = fixture["fixture"]["id"]
            newFixture.id_season = idSeason
            newFixture.date = dataFixtureFormatada
            newFixture.round = fixture["league"]["round"]
            newFixture.status = fixture["fixture"]["status"]["short"]
            newFixture.time_elapsed = fixture["fixture"]["status"]["elapsed"]

            newFixture.id = self.salvar(data=[newFixture]).getID()
            self.fixturesTeamsModel.atualizarDBFixturesTeams(data_fixture_api=fixture, id_fixture=newFixture.id)


    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS {self.name_table} (
            `id` INT NOT NULL AUTO_INCREMENT,
            `id_api` INT NOT NULL,
            `id_season` INT NOT NULL,
            `date` DATETIME NULL,
            `round` VARCHAR(255) NOT NULL,
            `status` VARCHAR(255) NOT NULL,
            `time_elapsed` VARCHAR(255) NULL,
            `last_get_statistics_api` DATETIME NULL,
            `last_get_lineups_api` DATETIME NULL,
            `last_modification` DATETIME NOT NULL,
                PRIMARY KEY (`id`),
                INDEX `id_season_fls_sea_idx` (`id_season` ASC) VISIBLE,
                CONSTRAINT `id_season_fls_sea`
                FOREIGN KEY (`id_season`)
                REFERENCES `season` (`id`)
                ON DELETE RESTRICT
                ON UPDATE RESTRICT,
                UNIQUE (`id_api`));"""

        self.executarQuery(query=query, params=[])


    def atualizarDados(self, id_season: int = None, arr_ids_team: list[int] = [], qtde_dados_estatisticas: int = 15):
        dateNow = datetime.now().strftime("%Y-%m-%d")

        if id_season is not None:
            season: Season = self.seasonsModel.obterByColumnsID(arrDados=[id_season])[0]
            functionAttFixtures = lambda: self.atualizarDBFixtures(idSeason=season.id)

            if season.last_get_fixtures_api is None or (season.last_get_fixtures_api.strftime("%Y-%m-%d") < dateNow and season.current == 1):
                self.atualizarTabela(model=self, functionAtualizacao=functionAttFixtures, isForçarAtualização=True)
                season.last_get_fixtures_api = datetime.now().strftime(self.seasonsModel.formato_datetime_YYYY_MM_DD_H_M_S)
                self.seasonsModel.salvar(data=season)


        if len(arr_ids_team) >= 1:
            for id_team in arr_ids_team:
                if id_team is None:
                    continue

                team: Team = self.teamsModel.obterByColumnsID(arrDados=[id_team])[0]

                if team.last_get_data_api is None or (team.last_get_data_api.strftime("%Y-%m-%d") < dateNow):
                    arrDataLeagues = self.leaguesModel.fazerConsultaApiFootball(id_team=team.id_api)

                    for dataLeague in arrDataLeagues:
                        leagueAPI = dataLeague["league"]
                        countryAPI = dataLeague["country"]
                        id_league_api = leagueAPI["id"]
                        arrLeaguesDB = self.leaguesModel.obterByReferenceApi(dadosBusca=[id_league_api])

                        if len(arrLeaguesDB) == 0:
                            country: Country = self.countriesModel.obterByColumns(arrNameColuns=["name"],
                                                                                  arrDados=[countryAPI["name"]])[0]
                            self.leaguesModel.atualizarDados(id_country=country.id)
                            arrLeaguesDB = self.obterByReferenceApi(dadosBusca=[id_league_api])

                        elif len(arrLeaguesDB) >= 2 or len(arrLeaguesDB) == 0:
                            raise "Leagues duplicadas ou sem league"

                        else:
                            league: League = arrLeaguesDB[0]
                            arrSeasons: list[Season] = self.seasonsModel.obterByColumns(arrNameColuns=["id_league"], arrDados=[league.id])

                            for season in arrSeasons:
                                arrFixturesEmAberto = self.obterFixturesOrderDataBy(id_season=season.id, id_team=None,
                                                                                    isApenasConcluidas=False,
                                                                                    isApenasEmAberto=True)
                                isPossuisFixturesEmAberto = len(arrFixturesEmAberto) >= 1

                                isObterFixtures = season.last_get_fixtures_api is None or \
                                                  (season.last_get_fixtures_api.strftime("%Y-%m-%d") < dateNow and season.current == 1) or \
                                                  (isPossuisFixturesEmAberto and season.current == 0)


                                if season.last_get_teams_api is None or (season.last_get_teams_api.strftime("%Y-%m-%d") < dateNow and season.current == 1):
                                    self.teamsModel.atualizarDados(id_season=season.id)
                                    season.last_get_teams_api = datetime.now().strftime(self.teamsModel.formato_datetime_YYYY_MM_DD_H_M_S)

                                if isObterFixtures:
                                    functionAttFixtures = lambda: self.atualizarDBFixtures(idSeason=season.id)
                                    self.atualizarTabela(model=self, functionAtualizacao=functionAttFixtures, isForçarAtualização=True)

                                    season.last_get_fixtures_api = datetime.now().strftime(self.formato_datetime_YYYY_MM_DD_H_M_S)

                                self.seasonsModel.salvar(data=[season])

                    team.last_get_data_api = datetime.now().strftime(self.teamsModel.formato_datetime_YYYY_MM_DD_H_M_S)
                    self.teamsModel.salvar(data=[team])

                arrFixturesForStatistics: list[Fixture] = self.obterFixturesOrderDataBy(id_team=id_team, isASC=False,
                                                                           limit=qtde_dados_estatisticas,
                                                                           isApenasConcluidas=True,
                                                                           isApenasComStatistics=True)

                #for fixture in arrFixturesForStatistics:
                    #seasonFixture: Season = self.seasonsModel.obterByColumnsID(arrDados=[fixture.id_season])[0]
                    #if fixture.last_get_statistics_api is None and seasonFixture.has_statistics_fixtures == 1:
                        #self.fixturesTeamsStatisticsModel.atualizarDados(id_fixture=fixture.id)

                        #fixture.last_get_statistics_api = datetime.now().strftime(self.formato_datetime_YYYY_MM_DD_H_M_S)
                        #self.salvar(data=[fixture])

    def obterFixturesOrderDataBy(self, id_season: int = None, id_team: int = None, isASC: bool = True, limit: int = None,
                                 isApenasConcluidas: bool = True, isApenasComStatistics: bool = False, isApenasEmAberto: bool = False) -> list[Fixture]:

        arrStatusConcluido = ["'FT'", "'AET'", "'PEN'"]
        arrStatusEmAberto = ["'NS'", "'TBD'", "'PST'"]
        queryOrder = " ORDER BY date" + (" ASC" if isASC else " DESC")
        queryLimite = f" LIMIT {limit}" if limit is not None else ""
        queryApenasConcluidas = " AND fix.status in (" + ",".join(arrStatusConcluido) + ")" if isApenasConcluidas else ""
        queryApenasComStatistics = " AND sea.has_statistics_fixtures = 1" if isApenasComStatistics else ""
        queryApenasEmAberto = " AND fix.status in (" + ",".join( arrStatusEmAberto) + ")" if isApenasEmAberto else ""
        arrParams = []

        if id_season is not None and id_team is not None:
            arrParams = [id_season, id_team]
            query = f"SELECT fix.* FROM {self.name_table} as fix " \
                    f" JOIN {self.fixturesTeamsModel.name_table} as fte on fte.id_fixture = fix.id" \
                    f" JOIN {self.seasonsModel.name_table} as sea on sea.id = fix.id_season" \
                    f" WHERE fix.id_season = %s AND fte.id_team = %s {queryApenasConcluidas} {queryApenasEmAberto} {queryApenasComStatistics}" \
                    f" {queryOrder} {queryLimite}"

        elif id_team is not None:
            arrParams = [id_team]
            query = f"SELECT fix.* FROM {self.name_table} as fix " \
                    f" JOIN {self.fixturesTeamsModel.name_table} as fte on fte.id_fixture = fix.id" \
                    f" JOIN {self.seasonsModel.name_table} as sea on sea.id = fix.id_season" \
                    f" WHERE fte.id_team = %s {queryApenasConcluidas} {queryApenasComStatistics}" \
                    f" {queryOrder} {queryLimite}"

        elif id_season is not None:
            arrParams = [id_season]
            query = f"SELECT fix.* FROM {self.name_table} as fix " \
                    f" JOIN {self.seasonsModel.name_table} as sea on sea.id = fix.id_season" \
                    f" WHERE fix.id_season = %s {queryApenasConcluidas} {queryApenasEmAberto} {queryApenasComStatistics}" \
                    f" {queryOrder} {queryLimite}"

        arrFixtures = self.database.executeSelectQuery(query=query, classModelDB=Fixture, params=arrParams)
        return arrFixtures

    def obterAllFixturesByIdsSeasons(self, arrIds: list[int]) -> list[Fixture]:
        arrIds = [str(id) for id in arrIds]
        query = f"SELECT * from {self.name_table} as fix WHERE fix.id_season in({','.join(arrIds)})" \
                f" AND (fix.status = 'FT' OR fix.status = 'AET' OR fix.status = 'PEN')" \
                f" ORDER BY fix.date ASC"

        arrFixtures = self.database.executeSelectQuery(query=query, classModelDB=Fixture)
        return arrFixtures

    def obterNextFixtureByidSeasonTeam(self, id_team: int, arrIdsFixtureIgnorar: list = [], id_season: int = None) -> list[Fixture]:
        sqlIdsIgnorar = ""
        arrIdsFixtureIgnorar = [str(idFix) for idFix in arrIdsFixtureIgnorar]
        if len(arrIdsFixtureIgnorar) >= 2:
            sqlIdsIgnorar = f" AND fix.id not in ({','.join(arrIdsFixtureIgnorar)})"
        elif len(arrIdsFixtureIgnorar) == 1:
            sqlIdsIgnorar = f" AND fix.id not in ({arrIdsFixtureIgnorar[0]})"

        if id_season is not None:
            sqlIdSeason = f"AND fix.id_season = {id_season}"
        else:
            sqlIdSeason = ""

        query = f"SELECT fix.* from {self.name_table} as fix" \
                f" JOIN fixture_teams as fte on fte.id_fixture = fix.id" \
                f" WHERE (fix.status <> 'FT' AND fix.status <> 'AET' AND fix.status <> 'PEN' AND fix.status <> 'CANC' " \
                f" AND fix.status <> 'PST' AND  fix.status <> 'WO' AND fix.status <> 'TBD')" \
                f" {sqlIdSeason} AND fte.id_team = {id_team}" \
                f" {sqlIdsIgnorar}" \
                f" ORDER BY fix.date ASC LIMIT 1"

        arrFixtures = self.database.executeSelectQuery(query=query, classModelDB=Fixture)
        return arrFixtures


class Fixture(ClassModel):
    def __init__(self, fixture: dict|object = None):
        self.id: int = None
        self.id_api: int = None
        self.id_season: int = None
        self.date: str = None
        self.round: str = None
        self.status: str = None
        self.time_elapsed: str = None
        self.last_get_statistics_api: str = None
        self.last_get_lineups_api: str = None
        self.last_modification: str = None

        super().__init__(dado=fixture)

