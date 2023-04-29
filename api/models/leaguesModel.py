from __future__ import annotations
from datetime import datetime
from api.models.model import Model, IdTabelas, ReferenciaDatabaseToAPI, ReferenciaTabelasFilhas, ReferenciaTabelasPai, ClassModel
from api.models.countriesModel import CountriesModel, Country
from api.models.seasonsModel import SeasonsModel, Season

class LeaguesModel(Model):
    def __init__(self):
        super().__init__(
            name_table="league",
            id_tabela=IdTabelas().league,
            name_columns_id=["id"],
            reference_db_api=[ReferenciaDatabaseToAPI(nome_coluna_db="id_api", nome_coluna_api="id")],
            referencia_tabelas_filhas=[ReferenciaTabelasFilhas(IdTabelas().season,
                                                               nome_tabela_filha="season",
                                                               nome_coluna_tabela_pai="id",
                                                               nome_coluna_tabela_filha="id_league")],
            referencia_tabelas_pai=[ReferenciaTabelasPai(id_tabela_pai=IdTabelas().country,
                                                         nome_tabela_pai="country",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_country")],
            classModelDB=League,
            rate_refesh_table_in_ms=15552384000)

        self.countriesModel = CountriesModel()
        self.criarTableDataBase()
        self.seasonsModel = SeasonsModel()


    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS  {self.name_table} (
            `id` INT NOT NULL AUTO_INCREMENT,
            `id_api` INT NOT NULL,
            `id_country` INT NOT NULL,
            `name` VARCHAR(255) NOT NULL,
            `type` VARCHAR(255) NOT NULL,
            `logo` LONGTEXT NULL,
            `is_obter_dados` TINYINT NOT NULL DEFAULT 0,
            `last_modification` DATETIME NOT NULL,
            PRIMARY KEY (`id`),
            CONSTRAINT `id_country`
            FOREIGN KEY (`id_country`)
            REFERENCES `country` (`id`)
            ON DELETE RESTRICT
            ON UPDATE RESTRICT,
            UNIQUE (`id_api`));"""

        self.executarQuery(query=query, params=[])

    def obterByidCounty(self, idCountry: int) -> list[League]:
        arrleagues: list[League] = self.obterByColumns(arrNameColuns=["id_country"], arrDados=[idCountry])
        return arrleagues

    def fazerConsultaApiFootball(self, id_league: int = None, name: str = None, country: str = None,
                                       code_country: str = None, season: int = None, id_team: int = None, type: str = None,
                                       current: str = None, search: str = None, last: int = None) -> list[dict]:
        arrParams = []
        query = "leagues"
        nameColumnResponseData = "response"

        if id_league is not None:
            arrParams.append("id=" + str(id_league))
        if name is not None:
            arrParams.append("name=" + name)
        if country is not None:
            arrParams.append("country=" + country)
        if code_country is not None:
            arrParams.append("code=" + code_country)
        if season is not None:
            arrParams.append("season=" + season)
        if id_team is not None:
            arrParams.append("team=" + str(id_team))
        if type is not None:
            arrParams.append("type=" + type)
        if current is not None:
            arrParams.append("current=" + current)
        if search is not None:
            arrParams.append("search=" + search)
        if last is not None:
            arrParams.append("last=" + last)

        if len(arrParams) >= 1:
            query += "?" + "&".join(arrParams)

        response = self.regraApiFootBall.conecarAPIFootball(query)
        responseData = response[nameColumnResponseData]

        return responseData


    def atualizarDBLeague(self, name_country: int = None, id_league_api: int = None, year_season: int = None,
                          id_team_api: int = None) -> None:

        arrLeagues = self.fazerConsultaApiFootball(country=name_country, id_league=id_league_api, season=year_season,
                                                   id_team=id_team_api)

        for data in arrLeagues:
            dataLeague = data["league"]
            dataCountry = data["country"]
            dataSeasons = data["seasons"]

            newLeague = League()
            newLeague.id = self.obterIdByReferenceIdApi(idApi=dataLeague["id"])
            newLeague.id_api = dataLeague["id"]
            newLeague.name = dataLeague["name"]
            newLeague.type = dataLeague["type"]
            newLeague.logo = dataLeague["logo"]

            arrCountries = self.countriesModel.obterByReferenceApi(dadosBusca=[dataCountry["name"]])

            if len(arrCountries) >= 2:
                raise "retornando mais dados que o previsto: Dados:\n" + str(countrie)
            elif len(arrCountries) == 0:
                raise "Sem dados necesários para continuar a salvar a liga"
            else:
                newLeague.id_country = arrCountries[0].id

            idLeagueSalvo = self.salvar(newLeague).getID()

            if idLeagueSalvo is None:
                raise "Reveja, nao teve dados salvos ou atualizados nas  liguas"

            self.seasonsModel.atualizarDBSeasonsByLeague(dataSeasons, idLeagueSalvo)

    def atualizarDados(self):
        arrCountries: list[Country] = self.countriesModel.obterTudo()

        for country in arrCountries:
            if country.is_obter_dados == 0:
                continue

            arrLeagues = self.obterByidCounty(idCountry=country.id)
            functionAttDB = lambda: self.atualizarDBLeague(name_country=country.name)

            if len(arrLeagues) == 0:
                self.atualizarTabela(model=self, functionAtualizacao=functionAttDB, isForçarAtualização=True)
            else:
                for league in arrLeagues:
                    seasonAtual = self.seasonsModel.obterSeasonAtualByIdLeague(idLeague=league.id)
                    dateNow = datetime.now().strftime("%Y-%m-%d")

                    if seasonAtual is None:
                        self.atualizarTabela(model=self, functionAtualizacao=functionAttDB)
                    elif seasonAtual.end is None:
                        raise "Season atual: " + str(seasonAtual.__dict__) + " está sem data de fim"
                    elif seasonAtual.end.strftime("%Y-%m-%d") < dateNow and seasonAtual.current == 1:
                        datetimeNow = datetime.now()
                        diff_ms = int((datetimeNow - seasonAtual.last_modification).total_seconds() * 1000)

                        if diff_ms >= self.rateRefeshTableInMs:
                            self.atualizarTabela(model=self, functionAtualizacao=functionAttDB, isForçarAtualização=True)
                    else:
                        continue

    def atualizarFlagIsObterDadosLeagueByTeam(self, id_team_api: int):
        arrDataLeagues = self.fazerConsultaApiFootball(id_team=id_team_api)

        for dataLeague in arrDataLeagues:
            leagueAPI = dataLeague["league"]
            countryAPI = dataLeague["country"]

            id_league_api = leagueAPI["id"]
            arrLeaguesDB = self.obterByReferenceApi(dadosBusca=[id_league_api])

            if len(arrLeaguesDB) == 0:
                self.atualizarDBLeague(name_country=countryAPI["name"])
                arrLeaguesDB = self.obterByReferenceApi(dadosBusca=[id_league_api])
            if len(arrLeaguesDB) >= 2 or len(arrLeaguesDB) == 0:
                print(len(arrLeaguesDB))
                raise "Leagues duplicadas ou sem no metrodo atualizarFlagIsObterDadosLeagueByTeam()"

            league: League = arrLeaguesDB[0]
            league.is_obter_dados = 1
            self.salvar(data=[league])

        self.atualizarDados()

    def atualizarSeasonsByIdLeague(self, id_league: int):
        league: League = self.obterByColumnsID(arrDados=[id_league])[0]
        league.last_modification: datetime = league.last_modification

        if league.last_modification.strftime("%Y-%m-%d") < datetime.now().strftime("%Y-%m-%d"):
            functionAttTeams = lambda: self.atualizarDBLeague(id_league_api=league.id_api)
            self.atualizarTabela(model=self, functionAtualizacao=functionAttTeams, isForçarAtualização=True)

            league.last_modification = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.salvar(data=[league])


class League(ClassModel):
    def __init__(self, league: dict|object = None):
        self.id: int = None
        self.id_api: int = None
        self.id_country: int = None
        self.name: str = None
        self.type: str = None
        self.logo: str = None
        self.is_obter_dados: int = None
        self.last_modification: str = None

        super().__init__(dado=league)