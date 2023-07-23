from __future__ import annotations
from api.models.fixturesModel import FixturesModel, Fixture
from api.models.fixturesTeamsModel import FixtureTeams
from api.models.teamsModel import TeamsModel, Team
from api.models.teamsSeasonsModel import TeamSeason
from api.models.seasonsModel import Season
from api.models.fixturesTeamsStatisticsModel import FixtureTeamStatistic

class FixturesRegras:
    def __init__(self):
        self.teamsModel = TeamsModel()
        self.fixturesModel = FixturesModel(teamsModel=self.teamsModel)

    def obter(self, id_season: int = None, id_team: int = None, isAsc: bool = True, limit: int = None,
              isApenasConcluidas: bool = True, isApenasComStatistics: bool = False, isObterProximoJogo: bool = False) -> list[Fixture]:

        arrFixtures: list[Fixture] = self.fixturesModel.obterFixturesOrderDataBy(id_season=id_season, id_team=id_team,
                                                                                 isASC=isAsc, limit=limit,
                                                                                 isApenasConcluidas=isApenasConcluidas,
                                                                                 isApenasComStatistics=isApenasComStatistics)

        for fixture in arrFixtures:
            fixture.teams = self.fixturesModel.fixturesTeamsModel.obterByColumns(arrNameColuns=["id_fixture"],
                                                                                 arrDados=[fixture.id])

        if isObterProximoJogo:
            fixture = self.obterProximoJogo(id_team=id_team)

            if fixture is not None:
                if isAsc:
                    arrFixtures.append(fixture)
                else:
                    arrFixtures.insert(0, fixture)

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
            fixture.teams: list[FixtureTeams] = self.fixturesModel.fixturesTeamsModel.obterByColumns(arrNameColuns=["id_fixture"],
                                                                                 arrDados=[fixture.id])

            fixture.season: Season = self.teamsModel.seasonsModel.obterByColumnsID(arrDados=[fixture.id_season])[0]

        if id_season is not None:
            isEncontrouNextPartida = False
            arrIdsFixtureIgnorar = []
            while not isEncontrouNextPartida:

                nextFixtureTeamHome = self.fixturesModel.obterNextFixtureByidSeasonTeam(id_season=None,
                                                                                        id_team=idTeamHome,
                                                                                        arrIdsFixtureIgnorar=arrIdsFixtureIgnorar)[0]

                nextFixtureTeamHome.teams = self.fixturesModel.fixturesTeamsModel.obterByColumns(arrNameColuns=["id_fixture"],
                                                                                                 arrDados=[nextFixtureTeamHome.id])

                if idTeamAway is not None:
                    for fixtureTeam in nextFixtureTeamHome.teams:
                        fixtureTeam: FixtureTeams = fixtureTeam
                        if fixtureTeam.id_team == idTeamAway:
                            isEncontrouNextPartida = True
                            nextFixtureTeamHome.season: Season = self.teamsModel.seasonsModel.obterByColumnsID(arrDados=[nextFixtureTeamHome.id_season])[0]
                            arrFixtures.append(nextFixtureTeamHome)

                    if not isEncontrouNextPartida:
                        arrIdsFixtureIgnorar.append(nextFixtureTeamHome.id)
                else:
                    nextFixtureTeamHome.season: Season = self.teamsModel.seasonsModel.obterByColumnsID(arrDados=[nextFixtureTeamHome.id_season])[0]
                    arrFixtures.append(nextFixtureTeamHome)
                    isEncontrouNextPartida = True

        return arrFixtures

    def obterFixturesByIdsTeam(self, arrIdsTeam: list[int]) -> list[Fixture]:
        arrIdsSeason = []
        for idTeam in arrIdsTeam:
            arrTeamsSeasons: list[TeamSeason] = self.teamsModel.teamsSeasonsModel.obterByColumns(
                arrNameColuns=["id_team"],
                arrDados=[idTeam])

            for teamSeason in arrTeamsSeasons:
                if teamSeason.id_season not in arrIdsSeason:
                    arrIdsSeason.append(teamSeason.id_season)

        arrFixtures = self.fixturesModel.obterAllFixturesByIdsSeasons(arrIds=arrIdsSeason)

        for fixture in arrFixtures:
            fixture.teams: list[FixtureTeams] = self.fixturesModel.fixturesTeamsModel.obterByColumns(arrNameColuns=["id_fixture"],
                                                                                 arrDados=[fixture.id])

            fixture.season: Season = self.teamsModel.seasonsModel.obterByColumnsID(arrDados=[fixture.id_season])[0]


        arrIdsFixtureIgnorar = []
        arrIdsTeamIgnorar = []

        for idteam in arrIdsTeam:
            isEncontrouNextPartida = False

            if idteam in arrIdsTeamIgnorar:
                continue

            while not isEncontrouNextPartida:
                nextFixtureTeam = self.fixturesModel.obterNextFixtureByidSeasonTeam(
                    id_team=idteam, arrIdsFixtureIgnorar=arrIdsFixtureIgnorar)[0]

                nextFixtureTeam.teams: list[FixtureTeams] = self.fixturesModel.fixturesTeamsModel.obterByColumns(
                    arrNameColuns=["id_fixture"], arrDados=[nextFixtureTeam.id])

                for fixtureTeam in nextFixtureTeam.teams:
                    arrIdsTeamIgnorar.append(fixtureTeam.id_team)

                nextFixtureTeam.season: Season = self.teamsModel.seasonsModel.obterByColumnsID(
                    arrDados=[nextFixtureTeam.id_season])[0]

                arrIdsFixtureIgnorar.append(nextFixtureTeam.id)
                arrFixtures.append(nextFixtureTeam)
                isEncontrouNextPartida = True

        return arrFixtures

    def obterFixtureEstatisticasByIdFixtureIdTeam(self, id_fixture: int, id_team: int) -> list[FixtureTeamStatistic]:
        arrFixturesEstatisticas: list[FixtureTeamStatistic] = \
            self.fixturesModel.fixturesTeamsStatisticsModel.obterByColumns(arrNameColuns=["id_fixture", "id_team"],
                                                                           arrDados=[id_fixture, id_team])

        return arrFixturesEstatisticas

    def obterProximoJogo(self, id_team: int, id_team_away: int = None) -> Fixture:
        isEncontrouNextPartida = False
        arrIdsFixtureIgnorar = []
        nextFixtureTeam = None

        while not isEncontrouNextPartida:

            arrNextFixtureTeam = self.fixturesModel.obterNextFixtureByidSeasonTeam(id_season=None,
                                                                                id_team=id_team,
                                                                                arrIdsFixtureIgnorar=arrIdsFixtureIgnorar)

            if len(arrNextFixtureTeam) == 0:
                return None

            nextFixtureTeam = arrNextFixtureTeam[0]
            nextFixtureTeam.teams = self.fixturesModel.fixturesTeamsModel.obterByColumns(
                arrNameColuns=["id_fixture"],
                arrDados=[nextFixtureTeam.id])

            if id_team_away is not None:
                for fixtureTeam in nextFixtureTeam.teams:
                    fixtureTeam: FixtureTeams = fixtureTeam
                    if fixtureTeam.id_team == id_team_away:
                        isEncontrouNextPartida = True
                        nextFixtureTeam.season: Season = \
                            self.teamsModel.seasonsModel.obterByColumnsID(arrDados=[nextFixtureTeam.id_season])[0]

                if not isEncontrouNextPartida:
                    arrIdsFixtureIgnorar.append(nextFixtureTeam.id)
            else:
                nextFixtureTeam.season: Season = \
                    self.teamsModel.seasonsModel.obterByColumnsID(arrDados=[nextFixtureTeam.id_season])[0]
                isEncontrouNextPartida = True

        return nextFixtureTeam