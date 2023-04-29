from __future__ import annotations
from api.models.model import Model, ReferenciaTabelasFilhas, IdTabelas, ReferenciaTabelasPai, ClassModel

class FixturesTeamsStatisticsModel(Model):
    def __init__(self):
        super().__init__(
            name_table="fixture_team_statistics",
            id_tabela=IdTabelas().fixture_team_estatistics,
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
                                    ReferenciaTabelasPai(IdTabelas().type_fixture_team_statistic,
                                                         nome_tabela_pai="type_fixture_team_statistic",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_type_statistic")],
            referencia_tabelas_filhas=[],
            classModelDB=FixtureTeamStatistic,
            rate_refesh_table_in_ms=0)


    def fazerConsultaFixturesStatisticsApiFootball(self, id_fixture: int, id_team: int = None,
                                                   name_type: str = None):
        """
            name_type consultar https://www.api-football.com/documentation-v3#tag/Fixtures/operation/get-fixtures-statistics
        """
        arrParams = []
        query = "fixtures/statistics"
        nameColumnResponseData = "response"

        if id_fixture is not None:
            arrParams.append("fixture=" + str(id_fixture))
        if id_team is not None:
            arrParams.append("team=" + str(id_team))
        if name_type is not None:
            arrParams.append("type=" + name_type)

        if len(arrParams) >= 1:
            query += "?" + "&".join(arrParams)

        response = self.regraApiFootBall.conecarAPIFootball(query)
        responseData = response[nameColumnResponseData]

        return responseData


    def atualizarDBFixtureTeamStatistics(self, idFixtureAPI: int) -> None:
        arrResponse = self.fazerConsultaFixturesStatisticsApiFootball(id_fixture=idFixtureAPI)

        for response in arrResponse:
            arrDataStatistics = response["statistics"]

            for dataStatistic in arrDataStatistics:
                typeStatisctic: TypesFixturesTeamsStatistics = self.typesFixturesTeamsStatisticsModel.obterTypesFixturesTeamsStatistics(
                    name_type=dataStatistic["type"])

                team: Team = self.teamsModel.obterByReferenceApi(dadosBusca=[response["team"]["id"]])[0]
                fixtureDB: Fixture = self.fixturesModel.obterByReferenceApi(dadosBusca=[idFixtureAPI])[0]

                newFixtureTeamStatistic = FixtureTeamStatistic()
                newFixtureTeamStatistic.id_fixture = fixtureDB.id
                newFixtureTeamStatistic.id_team = team.id
                newFixtureTeamStatistic.id_type_statistic = typeStatisctic.id

                if type(dataStatistic["value"]) == str:
                    newFixtureTeamStatistic.value = int(dataStatistic["value"].rstrip("%"))
                else:
                    newFixtureTeamStatistic.value = dataStatistic["value"]

                self.fixturesTeamsStatisticsModel.salvar(data=[newFixtureTeamStatistic.getDict()])


    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS `fixture_team_statistics` (
            `id` INT NOT NULL AUTO_INCREMENT,
            `id_fixture` INT NOT NULL,
            `id_team` INT NOT NULL,
            `id_type_statistic` INT NOT NULL,
            `value` INT NULL,
            `last_modification` DATETIME NOT NULL,
                PRIMARY KEY (`id`),
                INDEX `id_fixture_fts_fix_idx` (`id_fixture` ASC) VISIBLE,
                INDEX `id_team_fts_tea_idx` (`id_team` ASC) VISIBLE,
                INDEX `id_type_statistic_fts_tfts_idx` (`id_type_statistic` ASC) VISIBLE,
                CONSTRAINT `id_fixture_fts_fix`
                FOREIGN KEY (`id_fixture`)
                REFERENCES `fixture` (`id`)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
                CONSTRAINT `id_team_fts_tea`
                FOREIGN KEY (`id_team`)
                REFERENCES `team` (`id`)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
                CONSTRAINT `id_type_statistic_fts_tfts`
                FOREIGN KEY (`id_type_statistic`)
                REFERENCES `type_fixture_team_statistic` (`id`)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
                UNIQUE(`id_fixture`, `id_team`, `id_type_statistic`));"""

        self.executarQuery(query=query, params=[])


class FixtureTeamStatistic(ClassModel):
    def __init__(self, fixtureTeamStatistic: dict|object = None):
        self.id: int = None
        self.id_fixture: int = None
        self.id_team: int = None
        self.id_type_statistic: int = None
        self.value: int = None
        self.last_modification: str = None

        super().__init__(dado=fixtureTeamStatistic)