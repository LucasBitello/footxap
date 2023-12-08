from datetime import timedelta

from api.regras.fixturesRegras import FixturesRegras
from api.regras.leaguesSeasonsRegras import LeaguesRegras
from api.regras.teamsRegras import TeamsRegras
from api.regras.tabelaPontuacaoRegras import TabelaPontuacaoRegras, TeamPontuacao

from api.models.fixturesTeamsModel import FixtureTeams


class TeamJogo:
    def __init__(self):
        self.data_jogo: str = ""
        self.team_home: TeamPontuacao = TeamPontuacao()
        self.team_away: TeamPontuacao = TeamPontuacao()


class Tabelajogos:
    def __init__(self):
        self.arr_next_jogos: list[TeamJogo] = []


class TabelaJogosRegras:
    def __init__(self):
        self.fixturesRegras = FixturesRegras()
        self.leaguesRegras = LeaguesRegras()
        self.tabelaPontucaoRegras = TabelaPontuacaoRegras()
        self.teamsRegras = TeamsRegras()

    def obterTabelaJogos(self, id_season: int) -> Tabelajogos:
        tabelaPontuacao = self.tabelaPontucaoRegras.obterTabelaPontucao(id_season=id_season)
        newTabelaJogos = Tabelajogos()
        arrIdsFixtureIgnorar = []

        for team in tabelaPontuacao.arr_team_pontuacao:
            newTeamJogo = TeamJogo()
            nextFixture = self.fixturesRegras.fixturesModel.obterNextFixtureByidSeasonTeam(
                id_season=id_season,  id_team=team.id_team, arrIdsFixtureIgnorar=arrIdsFixtureIgnorar)

            if len(nextFixture) == 0:
                continue
            else:
                nextFixture = nextFixture[0]

            newTeamJogo.data_jogo = (nextFixture.date - timedelta(hours=3.0)).strftime("%Y-%m-%d %H:%M:%S")
            arrNextFixtureTeam: list[FixtureTeams] = (
                self.fixturesRegras.fixturesModel.fixturesTeamsModel.obterByColumns(arrNameColuns=["id_fixture"],
                                                                                    arrDados=[nextFixture.id]))

            for fixtureTeam in arrNextFixtureTeam:
                if fixtureTeam.is_home == 1:
                    for team2 in tabelaPontuacao.arr_team_pontuacao:
                        if team2.id_team == fixtureTeam.id_team:
                            newTeamJogo.team_home = team2
                elif fixtureTeam.is_home == 0:
                    for team2 in tabelaPontuacao.arr_team_pontuacao:
                        if team2.id_team == fixtureTeam.id_team:
                            newTeamJogo.team_away = team2

            newTabelaJogos.arr_next_jogos.append(newTeamJogo)
            arrIdsFixtureIgnorar.append(nextFixture.id)

        newTabelaJogos.arr_next_jogos = sorted(newTabelaJogos.arr_next_jogos,
                                               key=lambda teamJogos: teamJogos.data_jogo,
                                               reverse=False)
        return newTabelaJogos
