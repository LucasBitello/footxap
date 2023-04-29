from __future__ import annotations
from api.models.model import Model, ReferenciaTabelasFilhas, IdTabelas, ClassModel, ReferenciaTabelasPai

class FixturesTeamsLineupsModel(Model):
    def __init__(self):
        super().__init__(
            name_table="fixture_team_lineups",
            id_tabela=IdTabelas().fixture_team_lineups,
            name_columns_id=["id"],
            reference_db_api=[],
            referencia_tabelas_pai=[ReferenciaTabelasPai(IdTabelas().fixture,
                                                         nome_tabela_pai="fixture",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_fixture"),
                                    ReferenciaTabelasPai(IdTabelas().team,
                                                         nome_tabela_pai="team",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_team"),
                                    ReferenciaTabelasPai(IdTabelas().type_fixture_team_lineup,
                                                         nome_tabela_filha="type_fixture_team_lineup",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_type_lineup")],
            referencia_tabelas_filhas=[],
            classModelDB=FixtureTeamLineup,
            rate_refesh_table_in_ms=0)


    def fazerConsultaFixturesLineupsApiFootball(self, id_fixture: int, id_team: int = None, id_player: int = None,
                                                name_type: str = None):
        """
            name_type consultar https://www.api-football.com/documentation-v3#tag/Fixtures/operation/get-fixtures-statistics
        """
        arrParams = []
        query = "fixtures/lineups"
        nameColumnResponseData = "response"

        if id_fixture is not None:
            arrParams.append("fixture=" + str(id_fixture))
        if id_team is not None:
            arrParams.append("team=" + str(id_team))
        if id_player is not None:
            arrParams.append("player=" + str(id_player))
        if name_type is not None:
            arrParams.append("type=" + name_type)

        if len(arrParams) >= 1:
            query += "?" + "&".join(arrParams)

        response = self.regraApiFootBall.conecarAPIFootball(query)
        responseData = response[nameColumnResponseData]

        return responseData


    def atualizarDBFixtureTeamLineups(self, idFixtureAPI: int) -> None:
        arrResponse = self.fazerConsultaFixturesLineupsApiFootball(id_fixture=idFixtureAPI)

        for response in arrResponse:
            dataTeam = response["team"]
            formation: str = response["formation"]

            if formation is None:
                continue

            fixtureDB: Fixture = self.fixturesModel.obterByReferenceApi(dadosBusca=[idFixtureAPI])[0]
            typeLineup: TypesFixturesTeamsLineup = self.typesFixturesTeamsLineupsModel.obterTypeFixtureTeamsLineup(
                name_lineup=formation)

            newFixtureTeamLineup = FixtureTeamLineup()
            newFixtureTeamLineup.id_fixture = fixtureDB.id
            newFixtureTeamLineup.id_team = self.teamsModel.obterIdByReferenceIdApi(idApi=dataTeam["id"])
            newFixtureTeamLineup.id_type_lineup = typeLineup.id

            self.fixturesTeamsLineupsModel.salvar(data=[newFixtureTeamLineup])


    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS {self.name_table} (
            `id` INT NOT NULL AUTO_INCREMENT,
            `id_fixture` INT NOT NULL,
            `id_team` INT NOT NULL,
            `id_type_lineup` INT NOT NULL,
            `last_modification` DATETIME NOT NULL,
                PRIMARY KEY (`id`),
                INDEX `id_fixture_ftl_fix_idx` (`id_fixture` ASC) VISIBLE,
                INDEX `id_team_ftl_fix_idx` (`id_team` ASC) VISIBLE,
                INDEX `id_type_lineup_idx` (`id_type_lineup` ASC) VISIBLE,
                CONSTRAINT `id_fixture_ftl_fix`
                FOREIGN KEY (`id_fixture`)
                REFERENCES `fixture` (`id`)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
                CONSTRAINT `id_team_ftl_fix`
                FOREIGN KEY (`id_team`)
                REFERENCES `team` (`id`)
                ON DELETE NO ACTION
                ON UPDATE NO ACTION,
                CONSTRAINT `id_type_lineup`
                FOREIGN KEY (`id_type_lineup`)
                REFERENCES `type_fixture_team_lineup` (`id`)
                ON DELETE NO ACTION
                ON UPDATE NO ACTION,
                UNIQUE(`id_fixture`, `id_team`, `id_type_lineup`));"""

        self.executarQuery(query=query, params=[])


class FixtureTeamLineup(ClassModel):
    def __init__(self, fixtureTeamLineup: dict|object = None):
        self.id: int = None
        self.id_fixture: int = None
        self.id_team: int = None
        self.id_type_lineup: int = None
        self.last_modification: str = None

        super().__init__(dado=fixtureTeamLineup)