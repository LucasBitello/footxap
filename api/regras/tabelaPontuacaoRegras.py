from api.regras.fixturesRegras import FixturesRegras
from api.regras.leaguesSeasonsRegras import LeaguesRegras, SeasonsRegras
from api.regras.teamsRegras import TeamsRegras


from api.models.fixturesModel import Fixture
from api.models.fixturesTeamsModel import FixtureTeams
from api.models.leaguesModel import League
from api.models.seasonsModel import Season
from api.models.teamsModel import Team


class InfoTeamJogo:
    def __init__(self):
        self.is_home: int = None
        self.is_winner: int = None
        self.gols_marcados: int = 0
        self.gols_sofridos: int = 0

class TeamPontuacao:
    def __init__(self):
        self.name_team: str = None
        self.id_team: int = None
        self.pontos: int = 0
        self.saldo_gols: int = 0
        self.qtde_jogos: int = 0
        self.qtde_vitorias: int = 0
        self.qtde_empates: int = 0
        self.qtde_derrotas: int = 0
        self.qtde_gols_marcados: int = 0
        self.qtde_gols_sofridos: int = 0
        self.info_team: Team = None

        self.qtde_resultados_ultimos_jogos: int = 5
        self.arr_resultados_ultimos_jogos: list[InfoTeamJogo] = []

class TabelaPontuacao:
    def __init__(self):
        self.name_league: str = None
        self.year_season: int = None
        self.id_season: int = None
        self.arr_team_pontuacao: list[TeamPontuacao] = []


class TabelaPontuacaoRegras:
    def __init__(self):
        self.fixturesRegras = FixturesRegras()
        self.leaguesRegras = LeaguesRegras()
        self.seasonRegras = SeasonsRegras()
        self.teamsRegras = TeamsRegras()


    def obterTabelaPontucao(self, id_season: int) -> TabelaPontuacao:
        arrTeamsSeason: list[Team] = self.teamsRegras.obter(id_season=id_season)
        season: Season = self.seasonRegras.obter(id=id_season)[0]
        league: League = self.leaguesRegras.obter(idLeague=season.id_league)[0]

        newTabelaPontuacao = TabelaPontuacao()
        newTabelaPontuacao.id_season = season.id
        newTabelaPontuacao.year_season = season.year
        newTabelaPontuacao.name_league = league.name

        for team in arrTeamsSeason:
            newTeamPontuacao = TeamPontuacao()
            newTeamPontuacao.id_team = team.id
            newTeamPontuacao.name_team = team.name
            newTeamPontuacao.info_team = self.teamsRegras.obter(id=team.id)[0]

            last_index_ultimos_jogos = 0

            arrFixtures: list[Fixture] = self.fixturesRegras.obter(id_team=team.id, id_season=id_season)

            for fixture in arrFixtures:
                fixture.teams: list[FixtureTeams] = fixture.teams
                newUltimoJogo = InfoTeamJogo()

                for fixtureTeam in fixture.teams:
                    if fixtureTeam.id_team == team.id:
                        newUltimoJogo.is_home = fixtureTeam.is_home
                        newUltimoJogo.is_winner = fixtureTeam.is_winner

                        newTeamPontuacao.saldo_gols += fixtureTeam.goals
                        newTeamPontuacao.qtde_gols_marcados += fixtureTeam.goals


                        if fixtureTeam.is_winner == 0:
                            newTeamPontuacao.pontos += 0
                            newTeamPontuacao.qtde_derrotas += 1
                        elif fixtureTeam.is_winner == 1:
                            newTeamPontuacao.pontos += 3
                            newTeamPontuacao.qtde_vitorias += 1
                        elif fixtureTeam.is_winner is None:
                            newTeamPontuacao.pontos += 1
                            newTeamPontuacao.qtde_empates += 1

                    else:
                        newUltimoJogo.gols_sofridos += fixtureTeam.goals
                        newTeamPontuacao.saldo_gols -= fixtureTeam.goals
                        newTeamPontuacao.qtde_gols_sofridos += fixtureTeam.goals

                newTeamPontuacao.arr_resultados_ultimos_jogos.append(newUltimoJogo)
                newTeamPontuacao.qtde_jogos += 1
            newTabelaPontuacao.arr_team_pontuacao.append(newTeamPontuacao)

        newTabelaPontuacao.arr_team_pontuacao = sorted(newTabelaPontuacao.arr_team_pontuacao,
                                                       key=lambda teamPontuacao: (teamPontuacao.pontos,
                                                                                  teamPontuacao.qtde_jogos,
                                                                                  teamPontuacao.qtde_vitorias,
                                                                                  teamPontuacao.qtde_empates,
                                                                                  teamPontuacao.qtde_derrotas),
                                                       reverse=True)
        return newTabelaPontuacao