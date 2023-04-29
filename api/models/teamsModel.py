from __future__ import annotations
from datetime import datetime, timedelta

from api.models.model import Model, ReferenciaDatabaseToAPI, ReferenciaTabelasFilhas, ReferenciaTabelasPai, IdTabelas, ClassModel
from api.models.countriesModel import CountriesModel, Country
from api.models.teamsVenueModel import TeamsVenueModel, TeamVenue
from api.models.leaguesModel import LeaguesModel, League
from api.models.seasonsModel import SeasonsModel, Season
from api.models.teamsSeasonsModel import TeamsSeasonsModel, TeamSeason

class TeamsModel(Model):
    def __init__(self):
        super().__init__(
            name_table="team",
            id_tabela=IdTabelas().team,
            name_columns_id=["id"],
            reference_db_api=[ReferenciaDatabaseToAPI(nome_coluna_db="id_api", nome_coluna_api="id")],
            referencia_tabelas_filhas=[ReferenciaTabelasFilhas(id_tabela_filha=IdTabelas().team_venue,
                                                               nome_tabela_filha="team_venue",
                                                               nome_coluna_tabela_pai="id",
                                                               nome_coluna_tabela_filha="id_team"),
                                       ReferenciaTabelasFilhas(id_tabela_filha=IdTabelas().team_seasons,
                                                               nome_tabela_filha="team_seasons",
                                                               nome_coluna_tabela_pai="id",
                                                               nome_coluna_tabela_filha="id_team")],
            referencia_tabelas_pai=[ReferenciaTabelasPai(id_tabela_pai=IdTabelas().country,
                                                         nome_tabela_pai="country",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_country")],
            classModelDB=Team,
            rate_refesh_table_in_ms=60480)

        self.countriesModel = CountriesModel()
        self.leaguesModel = LeaguesModel()
        self.seasonsModel = SeasonsModel()
        self.criarTableDataBase()
        self.teamsVenueModel = TeamsVenueModel()
        self.teamsSeasonsModel = TeamsSeasonsModel()

    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS {self.name_table} (
            `id` INT NOT NULL AUTO_INCREMENT,
            `id_api` INT NOT NULL,
            `id_country` INT NOT NULL,
            `name` VARCHAR(45) NOT NULL,
            `code` VARCHAR(45) NULL,
            `founded` INT NULL,
            `national` INT NOT NULL,
            `logo` MEDIUMTEXT NULL,
            `last_modification` DATETIME NOT NULL,
                PRIMARY KEY (`id`),
                CONSTRAINT `id_country_tea_cou`
                FOREIGN KEY (`id_country`)
                REFERENCES `country` (`id`)
                ON DELETE RESTRICT
                ON UPDATE RESTRICT,
                UNIQUE (`id_api`));"""

        self.executarQuery(query=query, params=[])


    def fazerConsultaApiFootball(self, id_team: int = None, name: str = None, id_league: int = None,
                                       season: int = None, name_country: str = None, code_team: str = None,
                                       id_venue: str = None, search: str = None) -> list[dict]:
        arrParams = []
        query = "teams"
        nameColumnResponseData = "response"

        if id_team is not None:
            arrParams.append("id=" + id_team)
        if name is not None:
            arrParams.append("name=" + name)
        if id_league is not None:
            arrParams.append("league=" + str(id_league))
        if season is not None:
            arrParams.append("season=" + str(season))
        if name_country is not None:
            arrParams.append("country=" + name_country)
        if code_team is not None:
            arrParams.append("code=" + code_team)
        if id_venue is not None:
            arrParams.append("venue=" + id_venue)
        if search is not None:
            arrParams.append("search=" + search)

        if len(arrParams) >= 1:
            query += "?" + "&".join(arrParams)

        response = self.regraApiFootBall.conecarAPIFootball(query)
        responseData = response[nameColumnResponseData]

        return responseData


    def atualizarDBTeam(self, id_league_api: int, year_season: str) -> None:
        arrTeams = self.fazerConsultaApiFootball(id_league=id_league_api, season=year_season)
        id_league_db = self.leaguesModel.obterIdByReferenceIdApi(idApi=id_league_api)
        league: League = self.leaguesModel.obterByColumnsID(arrDados=[id_league_db])[0]
        country: Country = self.countriesModel.obterByColumnsID(arrDados=[league.id_country])[0]

        for data in arrTeams:
            dataTeam = data["team"]
            dataVenue = data["venue"]

            if country.name != dataTeam["country"] and country.name != "World":
                print("WARNING dados team está com o nome do pais errado: " + str(dataTeam))
                print(f"DB: {country.name}, API: {dataTeam['country']}")

            arrCountry: list[Country] = self.countriesModel.obterByReferenceApi(dadosBusca=[dataTeam["country"]])

            if len(arrCountry) == 0:
                arrCountry: list[Country] = self.countriesModel.obterByReferenceApi(dadosBusca=["World"])

            if len(arrCountry) == 0 or len(arrCountry) >= 2:
                print("team sem ou com muitos country: " + dataTeam["country"] + ", encontrado, dados team:" + str(dataTeam))
                raise

            country: Country = arrCountry[0]
            id_team_salvo: Team = self.obterIdByReferenceIdApi(dataTeam["id"])

            newTeam = Team()
            newTeam.id = id_team_salvo
            newTeam.id_country = country.id
            newTeam.id_api = dataTeam["id"]
            newTeam.name = dataTeam["name"]
            newTeam.code = dataTeam["code"]
            newTeam.logo = dataTeam["logo"]
            newTeam.founded = dataTeam["founded"]
            newTeam.national = dataTeam["national"]
            newTeam.last_modification = (datetime.now() - timedelta(days=2.0)).strftime("%Y-%m-%d %H:%M:%S")

            id_team_salvo = self.salvar(data=[newTeam]).getID()

            if dataVenue["id"] is not None:
                self.teamsVenueModel.atualizarDBTeamVenue(dataVenue, id_team_salvo)

            idLeague = self.leaguesModel.obterIdByReferenceIdApi(idApi=id_league_api)
            arrSeasons: list[Season] = self.seasonsModel.obterByColumns(arrNameColuns=["id_league", "year"],
                                                              arrDados=[idLeague, year_season])

            if len(arrSeasons) == 0 or len(arrSeasons) >= 2:
                raise "Muitas ou nenhuma seassons salva pra o team: " + str(dataTeam) + " e id_league: " + str(idLeague)

            self.teamsSeasonsModel.atualizarDBTeamSeason(id_team=id_team_salvo, id_season=arrSeasons[0].id)


    def atualizarDados(self, id_team: int = None):
        if id_team is not None:
            team: Team = self.obterByColumnsID(arrDados=[id_team])[0]
            self.leaguesModel.atualizarFlagIsObterDadosLeagueByTeam(id_team_api=team.id_api)

        arrSeasons: list[Season] = self.teamsSeasonsModel.obterSeasonsSemTeams()

        for season in arrSeasons:
            arrLeagues: list[League] = self.leaguesModel.obterByColumnsID(arrDados=[season.id_league])

            if len(arrLeagues) == 0 or len(arrLeagues) >= 2:
                raise "Opss parece que tem 2 ou nenhuma league para a mesma season " + str(season.__dict__)

            league: League = arrLeagues[0]
            functionAttTeams = lambda: self.atualizarDBTeam(id_league_api=league.id_api, year_season=season.year)
            self.atualizarTabela(model=self, functionAtualizacao=functionAttTeams, isForçarAtualização=True)


    def atualizarTeamsByLeagueSeason(self, id_season: int):
        season: Season = self.seasonsModel.obterByColumnsID(arrDados=[id_season])[0]
        league: League = self.leaguesModel.obterByColumnsID(arrDados=[season.id_league])[0]
        season.last_modification: datetime = season.last_modification

        print(season.last_modification)
        if season.last_modification.strftime("%Y-%m-%d") < datetime.now().strftime("%Y-%m-%d"):
            functionAttTeams = lambda: self.atualizarDBTeam(id_league_api=league.id_api, year_season=season.year)
            self.atualizarTabela(model=self, functionAtualizacao=functionAttTeams, isForçarAtualização=True)

            season.last_modification = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.seasonsModel.salvar(data=[season])

    def obterTeamsByName(self, name_team: str) -> list[Team]:
        query = "SELECT * FROM " + self.name_table + " where name like %s limit 15"
        paramNormalizado= '%' + name_team + '%'

        arrDados = self.executeSelectQuery(query=query, params=(paramNormalizado,))
        return arrDados


    def obterTeamsBySeasonName(self, name: str, id_season: int, id_team: int = None) -> list[Team]:
        query = f"SELECT tea.* FROM {self.name_table} as tea" \
                f" join team_seasons as tse on tse.id_team = tea.id" \
                f" where tea.name like '%{name}%' and tse.id_season = {id_season}"

        if id_team is not None:
            query += f" and tea.id = {id_team}"

        arrDados = self.database.executeSelectQuery(query=query, classModelDB=Team, params=[])
        return arrDados


class Team(ClassModel):
    def __init__(self, team: dict|object = None):
        self.id: int = None
        self.id_api: int = None
        self.id_country: int = None

        self.name: str = None
        self.code: str = None
        self.founded: int = None
        self.national: bool = None
        self.logo: str = None
        self.last_modification: str = None

        super().__init__(dado=team)