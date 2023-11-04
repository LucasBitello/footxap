from __future__ import annotations
from api.models.model import Model, IdTabelas, ClassModel, ReferenciaTabelasPai

class FixturesTeamsModel(Model):
    def __init__(self, teamsModel: object):
        self.teamsModel = teamsModel

        super().__init__(
            name_table="fixture_teams",
            id_tabela=IdTabelas().fixture_teams,
            name_columns_id=["id"],
            reference_db_api=[],
            referencia_tabelas_filhas=[],
            referencia_tabelas_pai=[ReferenciaTabelasPai(id_tabela_pai=IdTabelas().fixture,
                                                         nome_tabela_pai="fixture",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_fixture"),
                                    ReferenciaTabelasPai(id_tabela_pai=IdTabelas().fixture_teams,
                                                         nome_tabela_pai="team",
                                                         nome_coluna_tabela_pai="id",
                                                         nome_coluna_tabela_filha="id_team")],
            classModelDB=FixtureTeams,
            rate_refesh_table_in_ms=0)

        self.criarTableDataBase()

    def atualizarDBFixturesTeams(self, data_fixture_api: dict, id_fixture: int):
        id_team_home = self.teamsModel.obterIdByReferenceIdApi(data_fixture_api["teams"]["home"]["id"])
        id_team_away = self.teamsModel.obterIdByReferenceIdApi(data_fixture_api["teams"]["away"]["id"])

        if id_team_away is None or id_team_home is None:
            print(id_team_away, id_team_home)
            print("Um desses está nulo, rever : id: ", id_fixture, " dados: ", data_fixture_api)
            print("vai atualizar...")
            self.teamsModel.atualizarDBTeam(id_league_api=data_fixture_api["league"]["id"],
                                            year_season=data_fixture_api["league"]["season"])
            print("atualizou")
            id_team_home = self.teamsModel.obterIdByReferenceIdApi(data_fixture_api["teams"]["home"]["id"])
            id_team_away = self.teamsModel.obterIdByReferenceIdApi(data_fixture_api["teams"]["away"]["id"])

        newTeamHome = FixtureTeams()
        newTeamHome.id = self.obterIdByIdFixtureTeam(id_fixture=id_fixture, id_team=id_team_home)
        newTeamHome.id_fixture = id_fixture
        newTeamHome.id_team = id_team_home
        newTeamHome.is_winner = data_fixture_api["teams"]["home"]["winner"]
        newTeamHome.is_home = 1

        newTeamHome.goals = data_fixture_api["goals"]["home"]
        newTeamHome.goals_halftime = data_fixture_api["score"]["halftime"]["home"]
        newTeamHome.goals_fulltime = data_fixture_api["score"]["fulltime"]["home"]
        newTeamHome.goals_extratime = data_fixture_api["score"]["extratime"]["home"]
        newTeamHome.goals_penalty = data_fixture_api["score"]["penalty"]["home"]

        newTeamHome.goals_conceded = data_fixture_api["goals"]["away"]
        newTeamHome.goals_halftime_conceded = data_fixture_api["score"]["halftime"]["away"]
        newTeamHome.goals_fulltime_conceded = data_fixture_api["score"]["fulltime"]["away"]
        newTeamHome.goals_extratime_conceded = data_fixture_api["score"]["extratime"]["away"]
        newTeamHome.goals_penalty_conceded = data_fixture_api["score"]["penalty"]["away"]
        self.salvar(data=[newTeamHome])

        newTeamAway = FixtureTeams()
        newTeamAway.id = self.obterIdByIdFixtureTeam(id_fixture=id_fixture, id_team=id_team_away)
        newTeamAway.id_fixture = id_fixture
        newTeamAway.id_team = id_team_away
        newTeamAway.is_winner = data_fixture_api["teams"]["away"]["winner"]
        newTeamAway.is_home = 0

        newTeamAway.goals = data_fixture_api["goals"]["away"]
        newTeamAway.goals_halftime = data_fixture_api["score"]["halftime"]["away"]
        newTeamAway.goals_fulltime = data_fixture_api["score"]["fulltime"]["away"]
        newTeamAway.goals_extratime = data_fixture_api["score"]["extratime"]["away"]
        newTeamAway.goals_penalty = data_fixture_api["score"]["penalty"]["away"]

        newTeamAway.goals_conceded = data_fixture_api["goals"]["home"]
        newTeamAway.goals_halftime_conceded = data_fixture_api["score"]["halftime"]["home"]
        newTeamAway.goals_fulltime_conceded = data_fixture_api["score"]["fulltime"]["home"]
        newTeamAway.goals_extratime_conceded = data_fixture_api["score"]["extratime"]["home"]
        newTeamAway.goals_penalty_conceded = data_fixture_api["score"]["penalty"]["home"]


        self.salvar(data=[newTeamAway])


    def obterIdByIdFixtureTeam(self, id_fixture: int, id_team: int) -> int:
        arrFixturesTeams = self.obterByColumns(arrNameColuns=["id_fixture", "id_team"], arrDados=[id_fixture, id_team])

        if len(arrFixturesTeams) >= 2:
            raise f"""Há registros duplicados para a fixture_teams id_team: {id_team} id_fixture: {id_fixture}"""
        elif len(arrFixturesTeams) == 1:
            fixtureTeams: FixtureTeams = arrFixturesTeams[0]
            return fixtureTeams.id
        else:
            return None

    def criarTableDataBase(self):
        query = f"""CREATE TABLE IF NOT EXISTS {self.name_table} (
            `id` INT NOT NULL AUTO_INCREMENT,
            `id_fixture` INT NOT NULL,
            `id_team` INT NOT NULL,
            `is_winner` TINYINT(1) NULL,
            `is_home` TINYINT(1) NOT NULL,
            `goals` INT NOT NULL DEFAULT 0,
            `goals_halftime` INT NOT NULL DEFAULT 0,
            `goals_fulltime` INT NOT NULL DEFAULT 0,
            `goals_extratime` INT NOT NULL DEFAULT 0,
            `goals_penalty` INT NOT NULL DEFAULT 0,
            `goals_conceded` INT NOT NULL DEFAULT 0,
            `goals_halftime_conceded` INT NOT NULL DEFAULT 0,
            `goals_fulltime_conceded` INT NOT NULL DEFAULT 0,
            `goals_extratime_conceded` INT NOT NULL DEFAULT 0,
            `goals_penalty_conceded` INT NOT NULL DEFAULT 0,
            `last_modification` DATETIME NOT NULL,
                PRIMARY KEY (`id`),
                INDEX `id_team_fte_fix_idx` (`id_team` ASC) VISIBLE,
                CONSTRAINT `id_fixture_fte_fix`
                FOREIGN KEY (`id_fixture`)
                REFERENCES `fixture` (`id`)
                ON DELETE NO ACTION
                ON UPDATE NO ACTION,
                CONSTRAINT `id_team_fte_fix`
                FOREIGN KEY (`id_team`)
                REFERENCES `team` (`id`)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
                UNIQUE (`id_fixture`, `id_team`));"""

        self.executarQuery(query=query, params=[])


class FixtureTeams(ClassModel):
    def __init__(self, fixtureTeams = None):
        self.id: int = None
        self.id_fixture: int = None
        self.id_team: int = None
        self.is_winner: int = None
        self.is_home: int = None
        self.goals: int = 0
        self.goals_halftime: int = 0
        self.goals_fulltime: int = 0
        self.goals_extratime: int = 0
        self.goals_penalty: int = 0
        self.goals_conceded: int = 0
        self.goals_halftime_conceded: int = 0
        self.goals_fulltime_conceded: int = 0
        self.goals_extratime_conceded: int = 0
        self.goals_penalty_conceded: int = 0
        self.last_modification: str = None

        super().__init__(dado=fixtureTeams)

