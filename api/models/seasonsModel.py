from __future__ import annotations

from datetime import datetime, timedelta

from api.models.model import Model, ReferenciaDatabaseToAPI, ReferenciaTabelasFilhas, ReferenciaTabelasPai, IdTabelas, ClassModel

class SeasonsModel(Model):
    def __init__(self):
        super().__init__(
            name_table="season",
            id_tabela=IdTabelas().season,
            name_columns_id=["id"],
            reference_db_api=[],
            referencia_tabelas_filhas=[ReferenciaTabelasFilhas(IdTabelas().team_seasons,
                                                               nome_tabela_filha="team_seasons",
                                                               nome_coluna_tabela_pai="id",
                                                               nome_coluna_tabela_filha="id_season")],
            referencia_tabelas_pai=[ReferenciaTabelasPai(id_tabela_pai=IdTabelas().league,
                                                         nome_tabela_pai="league",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_league")],
            classModelDB=Season,
            rate_refesh_table_in_ms=86400000)

        self.criarTableDataBase()


    def obterSeasonAtualByIdLeague(self, idLeague: int) -> Season | None:
        query = f"SELECT * FROM {self.name_table} WHERE id_league = {idLeague} AND current = 1"
        arrSeasons: list[Season] = self.database.executeSelectQuery(query=query, classModelDB=self.classModelDB, params=[])

        if len(arrSeasons) == 0:
            return None

        return arrSeasons[0]

    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS  {self.name_table} (
            `id` INT NOT NULL AUTO_INCREMENT,
            `id_league` INT NOT NULL,
            `year` INT NOT NULL,
            `start` DATE NULL,
            `end` DATE NULL,
            `current` INT NULL,
            `has_events` INT NOT NULL,
            `has_lineups` INT NOT NULL,
            `has_statistics_fixtures` INT NOT NULL,
            `has_statistics_players` INT NOT NULL,
            `has_players` INT NOT NULL,
            `has_predictions` INT NOT NULL,
            `has_odds` INT NOT NULL,
            `last_get_data_api` DATETIME NULL,
            `last_get_teams_api` DATETIME NULL,
            `last_get_fixtures_api` DATETIME NULL,
            `last_modification` DATETIME NOT NULL,
            PRIMARY KEY (`id`),
            CONSTRAINT `id_league`
            FOREIGN KEY (`id_league`)
            REFERENCES `league` (`id`)
            ON DELETE RESTRICT
            ON UPDATE RESTRICT,
            UNIQUE (`id_league`, `year`));"""

        self.executarQuery(query=query, params=[])


    def atualizarDBSeasonsByLeague(self, arrSeasonsAPI: list[dict], id_league_db: int):
        if type(arrSeasonsAPI) != list or id_league_db is None:
            raise "Opaa dados em formato diferente dos desejados para as seasons: \n" + str(arrSeasonsAPI)

        for dataSeason in arrSeasonsAPI:
            arrSeason: list[Season] = self.obterByColumns(arrNameColuns=["id_league", "year"],
                                                         arrDados=[id_league_db, dataSeason["year"]])

            newSeason = Season()

            if len(arrSeason) == 0:
                newSeason.id = None
            elif len(arrSeason) >= 2:
                msgError = "seasson year:" + str(dataSeason["year"]) + "league_id: " + str(id_league_db) + " está duplicado."
                print(msgError)
                raise msgError
            else:
                newSeason = arrSeason[0]

            #Feito para evitar dados muito antigos não é util por Hora.
            if int(dataSeason["year"]) <= 2020:
                 continue

            newSeason.id_league = id_league_db
            newSeason.year = dataSeason["year"]
            newSeason.start = dataSeason["start"]
            newSeason.end = dataSeason["end"]
            newSeason.current = int(dataSeason["current"])
            newSeason.has_events = int(dataSeason["coverage"]["fixtures"]["events"])
            newSeason.has_lineups = int(dataSeason["coverage"]["fixtures"]["lineups"])
            newSeason.has_statistics_fixtures = int(dataSeason["coverage"]["fixtures"]["statistics_fixtures"])
            newSeason.has_statistics_players = int(dataSeason["coverage"]["fixtures"]["statistics_players"])
            newSeason.has_players = int(dataSeason["coverage"]["players"])
            newSeason.has_predictions = int(dataSeason["coverage"]["predictions"])
            newSeason.has_odds = int(dataSeason["coverage"]["odds"])

            self.salvar(data=[newSeason])

class Season(ClassModel):
    def __init__(self, season: dict = None):
        self.id: int = None
        self.id_league: int = None
        self.year: int = None
        self.start: str = None
        self.end: str = None
        self.current = None
        self.has_events: int = None
        self.has_lineups: int = None
        self.has_statistics_fixtures: int = None
        self.has_statistics_players: int = None
        self.has_players: int = None
        self.has_predictions: int = None
        self.has_odds: int = None
        self.last_get_data_api = None
        self.last_get_teams_api = None
        self.last_get_fixtures_api = None
        self.last_modification = None

        super().__init__(dado=season)