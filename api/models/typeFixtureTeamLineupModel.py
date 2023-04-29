from __future__ import annotations
from api.models.model import Model, IdTabelas, ReferenciaTabelasFilhas, ClassModel

class TypesFixturesTeamsLineupsModel(Model):
    def __init__(self):
        super().__init__(
            name_table="type_fixture_team_lineup",
            id_tabela=IdTabelas().type_fixture_team_lineup,
            name_columns_id=["id"],
            reference_db_api=[],
            referencia_tabelas_filhas=[ReferenciaTabelasFilhas(id_tabela_filha=IdTabelas().fixture_team_lineups,
                                                               nome_tabela_filha="fixture_team_lineups",
                                                               nome_coluna_tabela_filha="id_type_lineup",
                                                               nome_coluna_tabela_pai="id")],
            referencia_tabelas_pai=[],
            classModelDB=TypesFixturesTeamsLineup,
            rate_refesh_table_in_ms=0)

        self.criarTableDataBase()


    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS `type_fixture_team_lineup` (
          `id` INT NOT NULL AUTO_INCREMENT,
          `name_lineup` VARCHAR(45) NOT NULL,
          PRIMARY KEY (`id`),
          UNIQUE(`name_lineup`));"""

        self.executarQuery(query=query, params=[])


    def obterTypeFixtureTeamsLineup(self, name_lineup: str) -> TypesFixturesTeamsLineup:
        arrDados = self.obterByColumns(arrNameColuns=["name_lineup"], arrDados=[name_lineup])

        if len(arrDados) == 0:
            newTypeLineup = TypesFixturesTeamsLineup()
            newTypeLineup.name_lineup = name_lineup
            newTypeLineup.id = self.salvar(data=[newTypeLineup]).getID()
            return newTypeLineup

        return arrDados[0]


class TypesFixturesTeamsLineup(ClassModel):
    def __init__(self, typesFixturesTeamsLineup: dict = None):
        self.id: int = None
        self.name_lineup: str = None

        super().__init__(dado=typesFixturesTeamsLineup)