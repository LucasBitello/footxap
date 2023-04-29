from __future__ import annotations
from api.models.model import Model, ReferenciaTabelasFilhas, ReferenciaDatabaseToAPI, IdTabelas, ClassModel

class TypesFixturesTeamsStatistics(ClassModel):
    def __init__(self, typesFixturesTeamsStatistics: dict|object = None):
        self.id: int = None
        self.name_statistic: str = None

        super().__init__(dado=typesFixturesTeamsStatistics)


class TypesFixturesTeamsStatisticsModel(Model):
    def __init__(self):
        super().__init__(
            name_table="type_fixture_team_statistic",
            id_tabela=IdTabelas().type_fixture_team_statistic,
            name_columns_id=["id"],
            reference_db_api=[],
            referencia_tabelas_filhas=[ReferenciaTabelasFilhas(IdTabelas().fixture_team_estatistics,
                                                               nome_tabela_filha="fixture_team_statistics",
                                                               nome_coluna_tabela_filha="id_type_statistic",
                                                               nome_coluna_tabela_pai="id")],
            referencia_tabelas_pai=[],
            classModelDB=TypesFixturesTeamsStatistics,
            rate_refesh_table_in_ms=0)

        self.criarTableDataBase()


    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS `type_fixture_team_statistic` (
            `id` INT NOT NULL AUTO_INCREMENT,
            `name_statistic` VARCHAR(255) NOT NULL,
            PRIMARY KEY (`id`),
            UNIQUE  (`name_statistic`));"""

        self.executarQuery(query=query, params=[])


    def obterTypesFixturesTeamsStatistics(self, name_type: str) -> TypesFixturesTeamsStatistics:
        arrTypeStatistics: list[TypesFixturesTeamsStatistics] = self.obterByColumns(
            arrNameColuns=["name_statistic"], arrDados=[name_type])

        if len(arrTypeStatistics) == 0:
            newTypeStatistics = TypesFixturesTeamsStatistics()
            newTypeStatistics.name_statistic = name_type
            newTypeStatistics.id = self.salvar(data=[newTypeStatistics]).getID()

            return newTypeStatistics
        else:
            return arrTypeStatistics[0]