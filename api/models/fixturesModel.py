from __future__ import annotations
from datetime import datetime

from api.models.model import Model, ReferenciaDatabaseToAPI, ReferenciaTabelasFilhas, IdTabelas, ReferenciaTabelasPai, ClassModel
from api.models.leaguesModel import LeaguesModel, League
from api.models.seasonsModel import SeasonsModel, Season
from api.models.fixturesTeamsModel import FixturesTeamsModel, FixtureTeams
from api.models.fixturesTeamsLineupsModel import FixturesTeamsLineupsModel, FixtureTeamLineup
from api.models.fixturesTeamsStatisticsModel import FixturesTeamsStatisticsModel, FixtureTeamStatistic
from api.models.teamsSeasonsModel import TeamsSeasonsModel, TeamSeason
from api.models.teamsModel import TeamsModel, Team

class FixturesModel(Model):
    def __init__(self, teamsModel: object):
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
        '''self.fixturesTeamsLineupsModel = FixturesTeamsLineupsModel()
        self.fixturesTeamsStatisticsModel = FixturesTeamsStatisticsModel()'''


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

        if league.id in [82, 117, 118]:
            return

        arrFixtures = self.fazerConsultaFixturesApiFootball(id_league=league.id_api, year_season=season.year)

        for fixture in arrFixtures:
            dataFixtureFormatada = datetime.fromisoformat(fixture["fixture"]["date"]).strftime(
                "%Y-%m-%d %H:%M:%S")

            newFixture = Fixture()
            newFixture.id = self.obterIdByReferenceIdApi(fixture["fixture"]["id"])
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


    def atualizarDados(self, id_season: int = None, id_team: int = None):
        arrSeasons = self.obterSeasonsComTeamsSemFixtures(id_season=id_season, id_team=id_team)

        for season in arrSeasons:
            arrFixtures: list[Fixture] = self.obterByColumns(arrNameColuns=["id_season"], arrDados=[season.id])
            functionAttFixtures = lambda: self.atualizarDBFixtures(idSeason=season.id)
            self.atualizarTabela(model=self, functionAtualizacao=functionAttFixtures, isForçarAtualização=True)

            """if len(arrFixtures) == 0:
                self.atualizarTabela(model=self, functionAtualizacao=functionAttFixtures, isForçarAtualização=True)
            else:
                isAllSeasonConcluida = True

                for fixture in arrFixtures:
                    if (fixture.status != "FT" and fixture.status != "AET" and fixture.status != "PEN"):
                        isAllSeasonConcluida = False

                    if (fixture.date.strftime("%Y-%m-%d %H:%M") <= datetime.now().strftime("%Y-%m-%d  %H:%M")) \
                            and (fixture.status != "FT" and fixture.status != "AET" and fixture.status != "PEN" and fixture.status != "CANC" ):

                        print(f"######### atualizando dados pois faltava informaçoes id_fixture: {fixture.id} ########")
                        isAllSeasonConcluida = False

                        self.atualizarTabela(model=self, functionAtualizacao=functionAttFixtures,
                                             isForçarAtualização=True)

                if isAllSeasonConcluida and season.current == 1:
                    self.atualizarTabela(model=self, functionAtualizacao=functionAttFixtures, isForçarAtualização=True)"""

        if id_team is not None:
            arrSeasons = self.obterSeasonsComFixtures(id_season=id_season, id_team=id_team)

            for season in arrSeasons:
                arrFixtures: list[Fixture] = self.obterFixturesOrderDataBy(id_team=id_team, id_season=season.id,
                                                                           isApenasConcluidas=False)

                functionAttFixtures = lambda: self.atualizarDBFixtures(idSeason=season.id)
                self.atualizarTabela(model=self, functionAtualizacao=functionAttFixtures, isForçarAtualização=True)

                """for fixture in arrFixtures:
                    functionAttFixtures = lambda: self.atualizarDBFixtures(idSeason=season.id)
                    if (fixture.date.strftime("%Y-%m-%d %H:%M") < datetime.now().strftime("%Y-%m-%d %H:%M")) \
                            and (fixture.status != "FT" and fixture.status != "AET" and fixture.status != "PEN" and fixture.status != "CANC")\
                            and fixture.last_modification.strftime("%Y-%m-%d %H:%M") <= datetime.now().strftime("%Y-%m-%d %H:%M"):

                        print(f"##### atualizando dados pois faltava informaçoes time id_fixture: {fixture.id} #####")
                        self.atualizarTabela(model=self, functionAtualizacao=functionAttFixtures,
                                             isForçarAtualização=True)
                        break
                    else:
                        self.atualizarTabela(model=self, functionAtualizacao=functionAttFixtures,
                                             isForçarAtualização=True)
                        break"""


    def obterSeasonsComTeamsSemFixtures(self, id_season: int = None, id_team: int = None) -> list[Season]:
        arrStrQueryWhere = []

        if id_season is None and id_team is None:
            raise "è obrigatorio o id_team ou o id_season para obter as fixtures para atualizar."

        if id_season is not None:
            arrStrQueryWhere.append(f"sea.id = {id_season}")
        if id_team is not None:
            arrStrQueryWhere.append(f"tse.id_team = {id_team}")

        queryWhere = " AND " + " AND ".join(arrStrQueryWhere) if len(arrStrQueryWhere) >= 1 else ""

        query = f"SELECT sea.* FROM {self.seasonsModel.name_table} as sea" \
                f" JOIN {self.teamsSeasonsModel.name_table} as tse on tse.id_season = sea.id" \
                f" LEFT JOIN {self.name_table} as fix on fix.id_season = sea.id" \
                f" WHERE fix.id IS NULL {queryWhere}" \
                f" GROUP BY sea.id ORDER BY sea.year ASC"

        arrSeasons: list[Season] = self.database.executeSelectQuery(query=query, classModelDB=Season, params=[])
        return arrSeasons

    def obterFixturesOrderDataBy(self, id_season: int = None, id_team: int = None, isASC: bool = True,
                                 limit: int = None, isApenasConcluidas: bool = True) -> list[Fixture]:

        queryOrder = " ORDER BY date" + (" ASC" if isASC else " DESC")
        queryLimite = f" LIMIT {limit}" if limit is not None else ""
        queryApenasConcluidas = " AND fix.time_elapsed IS NOT NULL" if isApenasConcluidas else ""
        arrParams = []

        if id_season is not None and id_team is not None:
            arrParams = [id_season, id_team]
            query = f"SELECT fix.* FROM {self.name_table} as fix " \
                    f" JOIN {self.fixturesTeamsModel.name_table} as fte on fte.id_fixture = fix.id" \
                    f" WHERE fix.id_season = %s AND fte.id_team = %s {queryApenasConcluidas}" \
                    f" {queryOrder} {queryLimite}"

        elif id_team is not None:
            arrParams = [id_team]
            query = f"SELECT fix.* FROM {self.name_table} as fix " \
                    f" JOIN {self.fixturesTeamsModel.name_table} as fte on fte.id_fixture = fix.id" \
                    f" WHERE fte.id_team = %s {queryApenasConcluidas}" \
                    f" {queryOrder} {queryLimite}"

        elif id_season is not None:
            arrParams = [id_season]
            query = f"SELECT fix.* FROM {self.name_table} as fix " \
                    f" WHERE fix.id_season = %s {queryApenasConcluidas}" \
                    f" {queryOrder} {queryLimite}"

        arrFixtures = self.database.executeSelectQuery(query=query, classModelDB=Fixture, params=arrParams)
        return arrFixtures

    def obterAllFixturesByIdsSeasons(self, arrIds: list[int]) -> list[Fixture]:
        arrIds = [str(id) for id in arrIds]
        query = f"SELECT * from {self.name_table} as fix WHERE fix.id_season in({','.join(arrIds)})" \
                f" AND (fix.status = 'FT' OR fix.status = 'AET' OR fix.status = 'PEN')" \
                f" AND fix.id_season not in(222, 223, 224)" \
                f" ORDER BY fix.date ASC"

        arrFixtures = self.database.executeSelectQuery(query=query, classModelDB=Fixture)
        return arrFixtures

    def obterNextFixtureByidSeasonTeam(self, id_season: int, id_team: int) -> list[Fixture]:
        query = f"SELECT fix.* from {self.name_table} as fix" \
                f" JOIN fixture_teams as fte on fte.id_fixture = fix.id" \
                f" WHERE (fix.status <> 'FT' AND fix.status <> 'AET' AND fix.status <> 'PEN' AND fix.status <> 'CANC' " \
                f" AND fix.status <> 'PST')" \
                f" AND fix.id_season not in(222, 223, 224) and fte.id_team = {id_team}" \
                f" ORDER BY fix.date ASC LIMIT 1"

        arrFixtures = self.database.executeSelectQuery(query=query, classModelDB=Fixture)
        return arrFixtures


    def obterSeasonsComFixtures(self, id_season: int = None, id_team: int = None) -> list[Season]:
        arrStrQueryWhere = []

        if id_season is None and id_team is None:
            raise "è obrigatorio o id_team ou o id_season para obter as fixtures para atualizar."

        if id_season is not None:
            arrStrQueryWhere.append(f"sea.id = {id_season}")
        if id_team is not None:
            arrStrQueryWhere.append(f"tse.id_team = {id_team}")

        queryWhere = " AND " + " AND ".join(arrStrQueryWhere) if len(arrStrQueryWhere) >= 1 else ""

        query = f"SELECT sea.* FROM {self.seasonsModel.name_table} as sea" \
                f" JOIN {self.teamsSeasonsModel.name_table} as tse on tse.id_season = sea.id" \
                f" LEFT JOIN {self.name_table} as fix on fix.id_season = sea.id" \
                f" WHERE fix.id IS NOT NULL {queryWhere}" \
                f" GROUP BY sea.id ORDER BY sea.year ASC"

        arrSeasons: list[Season] = self.database.executeSelectQuery(query=query, classModelDB=Season, params=[])
        return arrSeasons


    def atualizarFixturesByidTeam(self, id_team: int):
        self.teamsModel.atualizarDados(id_team=id_team)
        team: Team = self.teamsModel.obterByColumnsID(arrDados=[id_team])[0]
        arrTeamSeason: list[TeamSeason] = self.teamsSeasonsModel.obterByColumns(arrNameColuns=["id_team"], arrDados=[id_team])

        if team.last_modification.strftime("%Y-%m-%d") < datetime.now().strftime("%Y-%m-%d"):
            for teamSeason in arrTeamSeason:
                if teamSeason.last_modification.strftime("%Y-%m-%d") >= datetime.now().strftime("%Y-%m-%d"):
                    continue

                self.atualizarDados(id_season=teamSeason.id_season, id_team=id_team)

                teamSeason.last_modification = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.teamsSeasonsModel.salvar(data=[teamSeason])

            team.last_modification = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.teamsModel.salvar(data=[team])


class Fixture(ClassModel):
    def __init__(self, fixture: dict|object = None):
        self.id: int = None
        self.id_api: int = None
        self.id_season: int = None
        self.date: str = None
        self.round: str = None
        self.status: str = None
        self.time_elapsed: str = None
        self.last_modification: str = None

        super().__init__(dado=fixture)

