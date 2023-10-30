from __future__ import annotations
from api.models.model import Model, IdTabelas, ReferenciaTabelasFilhas, ReferenciaTabelasPai, ClassModel


class NextGamesModel(Model):
    def __init__(self):
        super().__init__(
            name_table="next_games",
            id_tabela=IdTabelas().country,
            name_columns_id=["id"],
            reference_db_api=[],
            referencia_tabelas_filhas=[ReferenciaTabelasFilhas(id_tabela_filha=IdTabelas().league,
                                                               nome_tabela_filha="league",
                                                               nome_coluna_tabela_pai="id",
                                                               nome_coluna_tabela_filha="id_country")],
            referencia_tabelas_pai=[ReferenciaTabelasPai(id_tabela_pai=IdTabelas().team,
                                                         nome_tabela_pai="team",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_team_home"),
                                    ReferenciaTabelasPai(id_tabela_pai=IdTabelas().team,
                                                         nome_tabela_pai="team",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_team_away"),
                                    ReferenciaTabelasPai(id_tabela_pai=IdTabelas().season,
                                                         nome_tabela_pai="season",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_season"),
                                    ReferenciaTabelasPai(id_tabela_pai=IdTabelas().fixture,
                                                         nome_tabela_pai="fixture",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_fixture")],
            classModelDB=NextGames,
            rate_refesh_table_in_ms=31536000000)

        self.criarTableDataBase()

    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS {self.name_table}  (
                  `id` INT NOT NULL AUTO_INCREMENT,
                  `id_team_home` INT NOT NULL,
                  `id_team_away` INT NOT NULL,
                  `id_season` INT NOT NULL,
                  `id_fixture` INT NOT NULL,
                  `is_previu` TINYINT NOT NULL DEFAULT 0,
                  `is_acertou` TINYINT NOT NULL DEFAULT 0,
                  `data_jogo` DATETIME NOT NULL,
                  `last_modification` DATETIME NOT NULL,
                  PRIMARY KEY (`id`),
                  UNIQUE(`id_fixture`));"""

        self.executarQuery(query=query, params=[])


class NextGames(ClassModel):
    def __init__(self, nextGames: dict | object = None):
        self.id: int = None
        self.id_team_home: int = None
        self.id_team_away: int = None
        self.id_season: int = None
        self.id_fixture: int = None
        self.is_previu: int = None
        self.is_acertou: int = None
        self.data_jogo: str = None
        self.last_modification: str = None

        super().__init__(nextGames)
