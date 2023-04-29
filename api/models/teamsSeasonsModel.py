from __future__ import annotations
from datetime import datetime, timedelta

from api.models.model import Model, ReferenciaTabelasFilhas, IdTabelas, ClassModel, ReferenciaTabelasPai
from api.models.seasonsModel import SeasonsModel, Season
from api.models.leaguesModel import LeaguesModel, League

class TeamsSeasonsModel(Model):
    def __init__(self):
        self.leaguesModel = LeaguesModel()
        self.seasonsModel = SeasonsModel()

        super().__init__(
            name_table="team_seasons",
            id_tabela=IdTabelas().team_seasons,
            name_columns_id=["id"],
            reference_db_api=[],
            referencia_tabelas_filhas=[],
            referencia_tabelas_pai=[ReferenciaTabelasPai(id_tabela_pai=IdTabelas().team,
                                                                      nome_tabela_pai="team",
                                                                      nome_coluna_tabela_pai="id",
                                                                      nome_coluna_tabela_filha="id_team"),
                                    ReferenciaTabelasPai(id_tabela_pai=IdTabelas().season,
                                                                      nome_tabela_pai="season",
                                                                      nome_coluna_tabela_pai="id",
                                                                      nome_coluna_tabela_filha="id_season")],
            classModelDB=TeamSeason,
            rate_refesh_table_in_ms=0)

        self.criarTableDataBase()


    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS {self.name_table} (
            `id` INT NOT NULL AUTO_INCREMENT,
            `id_team` INT NOT NULL,
            `id_season` INT NOT NULL,
            `last_modification` DATETIME NOT NULL,
            PRIMARY KEY (`id`),
            INDEX `id_team_tls_tea_idx` (`id_team` ASC) VISIBLE,
            INDEX `id_team_tls_sle_idx` (`id_season` ASC) VISIBLE,
            CONSTRAINT `id_team_tls_tea`
            FOREIGN KEY (`id_team`)
            REFERENCES `team` (`id`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION,
            CONSTRAINT `id_team_tsl_sea`
            FOREIGN KEY (`id_season`)
            REFERENCES `season` (`id`)
            ON DELETE RESTRICT
            ON UPDATE RESTRICT,
            UNIQUE (`id_team`, `id_season`));"""

        self.executarQuery(query=query, params=[])

    def atualizarDBTeamSeason(self, id_team: int, id_season: int) -> None:
        arrTeamSeason: list[TeamSeason] = self.obterByColumns(arrNameColuns=["id_team", "id_season"],
                                                              arrDados=[id_team, id_season])
        newTeamSeasonLeague = TeamSeason()
        newTeamSeasonLeague.last_modification = (datetime.now() - timedelta(days=2.0)).strftime("%Y-%m-%d %H:%M:%S")

        if len(arrTeamSeason) >= 2:
            raise "Parere que esse time: " + str(id_team) + " e essa season: " + str(id_season) + " tem muitos registros"
        elif len(arrTeamSeason) == 1:
            newTeamSeasonLeague = arrTeamSeason[0]
            self.salvar(data=[newTeamSeasonLeague])
        else:
            newTeamSeasonLeague.id_team = id_team
            newTeamSeasonLeague.id_season = id_season
            self.salvar(data=[newTeamSeasonLeague])


    def obterSeasonsSemTeams(self) -> list[Season]:
        query = f"SELECT sea.* FROM {self.seasonsModel.name_table} as sea" \
                f" JOIN {self.leaguesModel.name_table} as lea on lea.id = sea.id_league" \
                f" LEFT JOIN {self.name_table} as tse on tse.id_season = sea.id" \
                f" WHERE tse.id IS NULL AND lea.is_obter_dados = 1" \
                f" GROUP BY sea.id_league, sea.year"

        arrDados: list[Season] = self.database.executeSelectQuery(query=query, classModelDB=Season)
        return arrDados


class TeamSeason(ClassModel):
    def __init__(self, teamSeason: dict|object = None):
        self.id: int = None
        self.id_team: int = None
        self.id_season: int = None
        self.last_modification: str = None

        super().__init__(dado=teamSeason)
