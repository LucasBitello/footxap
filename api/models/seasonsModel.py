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

    def obterByidLeague(self, idLeague: int) -> list[Season]:
        arrSeasons: list[Season] = self.obterByColumns(arrNameColuns=["id_league"], arrDados=[idLeague])
        return arrSeasons

    def obterSeasonAtualByIdLeague(self, idLeague: int):
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
                raise "seasson year:" + str(dataSeason["year"]) + "league_id: " + str(id_league_db) + " está duplicado."
            else:
                newSeason.id = arrSeason[0].id
                #Feito para evitar dados muito antigos não é util por Hora.
            if int(dataSeason["year"]) <= 2020:
                 continue

            newSeason.id_league = id_league_db
            newSeason.year = dataSeason["year"]
            newSeason.start = dataSeason["start"]
            newSeason.end = dataSeason["end"]
            newSeason.current = int(dataSeason["current"])
            newSeason.last_modification = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

            self.salvar(data=[newSeason])

class Season(ClassModel):
    def __init__(self, season: dict = None):
        self.id: int = None
        self.id_league: int = None
        self.year: int = None
        self.start: str = None
        self.end: str = None
        self.current = None
        self.last_modification = None

        super().__init__(dado=season)