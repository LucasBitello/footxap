from __future__ import annotations
from api.models.fixturesModel import FixturesModel, Fixture
from api.models.teamsModel import TeamsModel, Team
from api.models.teamsSeasonsModel import TeamSeason

class FixturesRegras:
    def __init__(self):
        self.teamsModel = TeamsModel()
        self.fixturesModel = FixturesModel(teamsModel=self.teamsModel)

    def obter(self, id_season: int = None, id_team: int = None) -> list[Fixture]:
        arrFixtures: list[Fixture] = self.fixturesModel.obterFixturesOrderDataBy(id_season=id_season, id_team=id_team)

        for fixture in arrFixtures:
            fixture.teams = self.fixturesModel.fixturesTeamsModel.obterByColumns(arrNameColuns=["id_fixture"],
                                                                                 arrDados=[fixture.id])
        return arrFixtures

    def obterTodasASFixturesSeasonAllTeamsByIdTeam(self, idTeamHome: int, idTeamAway: int = None, id_season: int = None) -> list[Fixture]:
        arrTeamHomeSeason: list[TeamSeason] = self.teamsModel.teamsSeasonsModel.obterByColumns(
            arrNameColuns=["id_team"],
            arrDados=[idTeamHome])

        arrIdsSeason = []

        for teamSeason in arrTeamHomeSeason:
            arrIdsSeason.append(teamSeason.id_season)

        if idTeamAway is not None:
            arrTeamAwaySeason: list[TeamSeason] = self.teamsModel.teamsSeasonsModel.obterByColumns(
                arrNameColuns=["id_team"],
                arrDados=[idTeamAway])

            for teamSeason in arrTeamAwaySeason:
                arrIdsSeason.append(teamSeason.id_season)

        arrFixtures = self.fixturesModel.obterAllFixturesByIdsSeasons(arrIds=arrIdsSeason)

        for fixture in arrFixtures:
            fixture.teams = self.fixturesModel.fixturesTeamsModel.obterByColumns(arrNameColuns=["id_fixture"],
                                                                                 arrDados=[fixture.id])

        if id_season is not None:
            nextFixtureTeamHome = self.fixturesModel.obterNextFixtureByidSeasonTeam(id_season=id_season, id_team=idTeamHome)[0]
            nextFixtureTeamHome.teams = self.fixturesModel.fixturesTeamsModel.obterByColumns(arrNameColuns=["id_fixture"],
                                                                                             arrDados=[nextFixtureTeamHome.id])
            arrFixtures.append(nextFixtureTeamHome)

            """if idTeamAway is not None:
                nextFixtureTeamAway = self.fixturesModel.obterNextFixtureByidSeasonTeam(id_season=id_season, id_team=idTeamAway)[0]
                nextFixtureTeamAway.teams = self.fixturesModel.fixturesTeamsModel.obterByColumns(arrNameColuns=["id_fixture"],
                                                                                                 arrDados=[nextFixtureTeamAway.id])
                arrFixtures.append(nextFixtureTeamAway)"""

        return arrFixtures