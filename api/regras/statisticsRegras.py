import numpy
from json import dumps
from datetime import datetime, timedelta
from operator import itemgetter
from matplotlib import pyplot as plt

from api.regras.teamsRegras import TeamsRegras
from api.regras.fixturesRegras import FixturesRegras
from api.regras.leaguesSeasonsRegras import SeasonsRegras, LeaguesRegras

from api.models.fixturesModel import Fixture
from api.models.fixturesTeamsModel import FixtureTeams
from api.models.seasonsModel import Season
from api.models.teamsModel import Team
from api.models.teamsSeasonsModel import TeamSeason

class TeamInfoDataSet:
    def __init__(self):
        self.is_prever: int = None
        self.data_fixture: int = None
        self.is_home: int = None
        self.id_ordem: int = None
        self.id_season: int = None
        self.is_winner: int = None

        self.id_team: int = None
        self.pontos_season: int = 0
        self.media_gols_marcados: int = None
        self.media_gols_sofridos: int = None

        self.id_adversario: int = None

        self.is_next_game_season_home: int = 2
        self.is_next_game_home: int = 2
        self.gols_marcados: int = None
        self.gols_sofridos: int = None

class TeamsPlaysSaida:
    def __init__(self):
        self.is_winner_home: int = None
        self.is_winner_away: int = None
        self.qtde_gols_marcados: int = None

class TeamsPlaysEntrada:
    def __init__(self):
        self.is_prever: int = None
        self.data_fixture: int = None
        self.id_season: int = None

        self.name_team_home: str = None
        self.id_team_home: int = None
        self.qtde_pontos_season_home: int = None
        self.qtde_saldo_gols_home: int = None
        self.media_gols_home: int = None
        self.qtde_gols_marcados_home: int = None

        self.name_team_away: str = None
        self.id_team_away: int = None
        self.qtde_pontos_season_away: int = None
        self.qtde_saldo_gols_away: int = None
        self.media_gols_away: int = None
        self.qtde_gols_marcados_away: int = None

        self.saida_prevista: TeamsPlaysSaida = TeamsPlaysSaida()


class TeamInfo:
    def __init__(self):
        self.id_team: int = None
        self.arrDatasFixtures: list[str] = []
        self.ultimosResultados: list[int] = []

        self.ultimosGolsFeitos: list[int] = []
        self.ultimosGolsSofridos: list[int] = []
        self.mediaUltimosGolsMarcados: int = 0
        self.mediaUltimosGolsSofridos: int = 0



class StatisticsRegras:
    def __init__(self):
        self.teamsRegras = TeamsRegras()
        self.seasonRegras = SeasonsRegras()
        self.fixturesRegras = FixturesRegras()

    def obter(self, id_season: int = None, id_team: int = None) -> list:
        if id_season is None and id_team is None and id_team_away is None:
            raise "Id_team ou id_season é obrigátorio para as estatisticas."

        arrTeams: list[Team] = self.teamsRegras.obter(id=id_team, id_season=id_season)
        arrTeamPontuacao: list = []
        qtdeUltimosJogosGols = 10
        for team in arrTeams:
            arrFixtures = self.fixturesRegras.obter(id_season=id_season, id_team=team.id)


    def obterCorelacaoEntretimes(self, id_team_home: int, id_team_away: int = None) -> None:
        if id_team_away is None:
            return

        arrFixturesTeamHome = self.fixturesRegras.obter(id_team=id_team_home)

        arrIdsTeamsEmfrentadosHome = []
        for fixture in arrFixturesTeamHome:
            fixture.is_encontrou_jogo = False
            for fixTeam in fixture.teams:
                fixTeam: FixtureTeams = fixTeam
                if fixTeam.id_team != id_team_home:
                    arrIdsTeamsEmfrentadosHome.append(fixTeam.id_team)

        arrFixturesTeamAway = self.fixturesRegras.obter(id_team=id_team_away)

        arrIdsTeamsEmfrentadosAway = []
        for fixture in arrFixturesTeamAway:
            fixture.is_encontrou_jogo = False
            for fixTeam in fixture.teams:
                fixTeam: FixtureTeams = fixTeam
                if fixTeam.id_team != id_team_away:
                    arrIdsTeamsEmfrentadosAway.append(fixTeam.id_team)

        arridsTeamEmComum = []
        arridsTeamNaoComum = []

        for idHome in arrIdsTeamsEmfrentadosHome:
            isEncontrouIgual = False
            for idAway in arrIdsTeamsEmfrentadosAway:
                if idHome == idAway:
                    isEncontrouIgual = True
                    try:
                        arridsTeamEmComum.index(idHome)
                    except:
                        arridsTeamEmComum.append(idHome)
            if not isEncontrouIgual:
                try:
                    arridsTeamNaoComum.index(idHome)
                except:
                    arridsTeamNaoComum.append(idHome)


        qtdeVitoriasHome = 0
        qtdeVitoriasAway = 0
        qtdeDerrotasHome = 0
        qtdeDerrotasAway = 0
        qtdeEmpatesHome = 0
        qtdeEmpatesAway = 0

        print(arridsTeamEmComum)
        print(arridsTeamNaoComum)
        for idTeam in arridsTeamEmComum:
            for fixtureHome in arrFixturesTeamHome:
                for fixTeam in fixtureHome.teams:
                    fixTeam: FixtureTeams = fixTeam
                    if fixTeam.id_team == idTeam and not fixtureHome.is_encontrou_jogo:
                        fixtureHome.is_encontrou_jogo = True
                        if fixTeam.is_winner == 1:
                            qtdeDerrotasHome += 1
                        elif fixTeam.is_winner == None:
                            qtdeEmpatesHome += 1
                        else:
                            qtdeVitoriasHome += 1

            for fixtureAway in arrFixturesTeamAway:
                for fixTeam in fixtureAway.teams:
                    fixTeam: FixtureTeams = fixTeam
                    if fixTeam.id_team == idTeam and not fixtureAway.is_encontrou_jogo:
                        fixtureAway.is_encontrou_jogo = True
                        if fixTeam.is_winner == 1:
                            qtdeDerrotasAway += 1
                        elif fixTeam.is_winner == None:
                            qtdeEmpatesAway += 1
                        else:
                            qtdeVitoriasAway += 1

        print(qtdeVitoriasHome, qtdeEmpatesHome, qtdeDerrotasHome)
        print(qtdeVitoriasAway, qtdeEmpatesAway, qtdeDerrotasAway)

    def obterDadosTodasSeasonLeaguesByTeam(self, idTeamHome: int, idTeamAway: int = None, id_season: int = None) -> list[TeamInfoDataSet]:
        arrKeysFinished = ['FT', 'AET', 'PEN', 'CANC']
        arrFixtures: list[Fixture] = self.fixturesRegras.obterTodasASFixturesSeasonAllTeamsByIdTeam(idTeamHome=idTeamHome,
                                                                                                    idTeamAway=idTeamAway,
                                                                                                    id_season=id_season)
        arrTeamInfoDataset: list[TeamInfoDataSet] = []
        lastIdsInfo = 0
        for fixture in arrFixtures:
            for indexTeamFixture in range(len(fixture.teams)):
                lastIdsInfo += 1
                teamFixture: FixtureTeams = fixture.teams[indexTeamFixture]
                newTeamInfoDataset = TeamInfoDataSet()

                newTeamInfoDataset.is_prever = 1 if fixture.status not in arrKeysFinished else 0
                newTeamInfoDataset.id_ordem = lastIdsInfo
                newTeamInfoDataset.data_fixture = fixture.date
                newTeamInfoDataset.id_season = fixture.id_season
                newTeamInfoDataset.id_team = teamFixture.id_team
                newTeamInfoDataset.gols_marcados = teamFixture.goals
                newTeamInfoDataset.is_home = teamFixture.is_home


                if indexTeamFixture == 0:
                    outherFixtureTeam: FixtureTeams = fixture.teams[1]
                else:
                    outherFixtureTeam: FixtureTeams = fixture.teams[0]

                newTeamInfoDataset.id_adversario = outherFixtureTeam.id_team
                newTeamInfoDataset.gols_sofridos = outherFixtureTeam.goals

                golsTotais = newTeamInfoDataset.gols_marcados - newTeamInfoDataset.gols_sofridos

                if golsTotais == 0:
                    newTeamInfoDataset.is_winner = 1
                elif golsTotais >= 1:
                    newTeamInfoDataset.is_winner = 2
                else:
                    newTeamInfoDataset.is_winner = 0

                ultimaTeamInfoSeason = self.obterUltimaInfoTeamArray(arrTeamInfoDataset=arrTeamInfoDataset,
                                                                     id_team=newTeamInfoDataset.id_team,
                                                                     id_season=newTeamInfoDataset.id_season)

                if ultimaTeamInfoSeason is not None:
                    if ultimaTeamInfoSeason.is_winner == 2:
                        newTeamInfoDataset.pontos_season = ultimaTeamInfoSeason.pontos_season + 3
                    elif ultimaTeamInfoSeason.is_winner == 1:
                        newTeamInfoDataset.pontos_season = ultimaTeamInfoSeason.pontos_season + 1
                    else:
                        newTeamInfoDataset.pontos_season = ultimaTeamInfoSeason.pontos_season + 0


                self.atualizarUltimaInfoTeamArray(arrTeamInfoDataset=arrTeamInfoDataset, id_team=teamFixture.id_team,
                                                  name_atributo="is_next_game_home",
                                                  value_atributo=newTeamInfoDataset.is_home)

                self.atualizarUltimaInfoTeamArray(arrTeamInfoDataset=arrTeamInfoDataset, id_team=teamFixture.id_team,
                                                  name_atributo="is_next_game_season_home",
                                                  value_atributo=newTeamInfoDataset.is_home,
                                                  id_season=newTeamInfoDataset.id_season)

                newTeamInfoDataset.media_gols_marcados, \
                    newTeamInfoDataset.media_gols_sofridos = self.calcularMediaGolsUltimosJogos(
                    arrTeamInfoDataset=arrTeamInfoDataset, id_team=newTeamInfoDataset.id_team, id_season=None)

                arrTeamInfoDataset.append(newTeamInfoDataset)

        return arrTeamInfoDataset

    def obterAllFixturesByIdTeams(self, idTeamHome: int, idTeamAway: int = None, id_season: int = None) -> list[TeamsPlaysEntrada]:
        arrKeysFinished = ['FT', 'AET', 'PEN', 'CANC']
        arrFixtures: list[Fixture] = self.fixturesRegras.obterTodasASFixturesSeasonAllTeamsByIdTeam(
            idTeamHome=idTeamHome,
            idTeamAway=idTeamAway,
            id_season=id_season)

        arrTeamsPlays: list[TeamsPlaysEntrada] = []

        for fixture in arrFixtures:
            newTeamPlays = TeamsPlaysEntrada()
            newTeamPlays.id_season = fixture.id_season
            newTeamPlays.data_fixture = fixture.date
            newTeamPlays.is_prever = 1 if fixture.status not in arrKeysFinished else 0
            newTeamPlays.saida_prevista.qtde_gols_marcados = 0

            fixture.teams: list[FixtureTeams] = fixture.teams
            indexOutherTeam = 1

            arrIdsHomeAway = [idTeamHome, idTeamAway]
            if fixture.teams[0].id_team not in arrIdsHomeAway and fixture.teams[1].id_team not in arrIdsHomeAway:
                continue

            if newTeamPlays.is_prever == 1:
                print("Vai prever o jogo para a data: ", fixture.date)
                dateFutura = (datetime.now() + timedelta(days=2.0)).strftime("%Y-%m-%d %H:%M:%S")

                if fixture.date.strftime("%Y-%m-%d %H:%M:%S") >= dateFutura or \
                        fixture.date.strftime("%Y-%m-%d %H:%M:%S") < datetime.now().strftime("%Y-%m-%d %H:%M:%S"):
                    print("DB não possui a fixture atual desejada: ", fixture.date, " : ", dateFutura)
                    raise "Erro sem fixture"

            for team in fixture.teams:
                if team.id_team == idTeamHome or (team.id_team != idTeamAway and fixture.teams[indexOutherTeam].id_team != idTeamHome):
                    newTeamPlays.id_team_home = team.id_team
                    newTeamPlays.name_team_home = self.teamsRegras.teamsModel.obterByColumnsID(arrDados=[team.id_team])[0].name

                    if team.is_winner == 0:
                        newTeamPlays.saida_prevista.is_winner_home = 0
                    elif team.is_winner is None:
                        newTeamPlays.saida_prevista.is_winner_home = 1
                    elif team.is_winner == 1:
                        newTeamPlays.saida_prevista.is_winner_home = 2

                    newTeamPlays.qtde_gols_marcados_home = team.goals
                    newTeamPlays.saida_prevista.qtde_gols_marcados += team.goals

                    ultimaTeamsPlaySeasonHome = self.obterUltimaTeamPlay(arrTeamsPlaysEntrada=arrTeamsPlays,
                                                                     id_team=newTeamPlays.id_team_home,
                                                                     id_season=newTeamPlays.id_season)

                    if ultimaTeamsPlaySeasonHome is None:
                        newTeamPlays.qtde_pontos_season_home = 0
                        newTeamPlays.qtde_saldo_gols_home = 0
                    else:
                        newTeamPlays.qtde_pontos_season_home = self.obterPontuacao(teamsPlays=ultimaTeamsPlaySeasonHome,
                                                                                   id_team=newTeamPlays.id_team_home)
                        newTeamPlays.qtde_saldo_gols_home = self.obterSaldoGols(teamsPlays=ultimaTeamsPlaySeasonHome,
                                                                                id_team=newTeamPlays.id_team_home)

                    newTeamPlays.media_gols_home = self.calcularMediaGolsTeamsPlay(arrTeamsPlaysEntrada=arrTeamsPlays,
                                                                                   id_team=newTeamPlays.id_team_home,
                                                                                   id_season=newTeamPlays.id_season)[0]


                else:
                    newTeamPlays.id_team_away = team.id_team
                    newTeamPlays.name_team_away = self.teamsRegras.teamsModel.obterByColumnsID(arrDados=[team.id_team])[0].name

                    if team.is_winner == 0:
                        newTeamPlays.saida_prevista.is_winner_away = 0
                    elif team.is_winner is None:
                        newTeamPlays.saida_prevista.is_winner_away = 1
                    elif team.is_winner == 1:
                        newTeamPlays.saida_prevista.is_winner_away = 2

                    newTeamPlays.qtde_gols_marcados_away = team.goals
                    newTeamPlays.saida_prevista.qtde_gols_marcados += team.goals

                    ultimaTeamsPlaySeasonAway = self.obterUltimaTeamPlay(arrTeamsPlaysEntrada=arrTeamsPlays,
                                                                         id_team=newTeamPlays.id_team_away,
                                                                         id_season=newTeamPlays.id_season)

                    if ultimaTeamsPlaySeasonAway is None:
                        newTeamPlays.qtde_pontos_season_away = 0
                        newTeamPlays.qtde_saldo_gols_away = 0
                    else:
                        newTeamPlays.qtde_pontos_season_away = self.obterPontuacao(teamsPlays=ultimaTeamsPlaySeasonAway,
                                                                                   id_team=newTeamPlays.id_team_away)
                        newTeamPlays.qtde_saldo_gols_away = self.obterSaldoGols(teamsPlays=ultimaTeamsPlaySeasonAway,
                                                                                id_team=newTeamPlays.id_team_away)

                    newTeamPlays.media_gols_away = self.calcularMediaGolsTeamsPlay(arrTeamsPlaysEntrada=arrTeamsPlays,
                                                                                   id_team=newTeamPlays.id_team_away,
                                                                                   id_season=newTeamPlays.id_season)[0]
                indexOutherTeam = 0

            arrTeamsPlays.append(newTeamPlays)
        return arrTeamsPlays

    def obterStatisticsByTeam(self, arrTeamsInfoDataset: list[TeamInfoDataSet], idTeam: int) -> TeamInfo:
        newTeamInfo = TeamInfo()
        nFor = -1

        for teamInfoDataset in list(reversed(arrTeamsInfoDataset)):
            if teamInfoDataset.id_team == idTeam and nFor <= 9:
                nFor += 1
                newTeamInfo.id_team = idTeam
                newTeamInfo.arrDatasFixtures.append(teamInfoDataset.data_fixture)
                newTeamInfo.ultimosGolsFeitos.append(teamInfoDataset.gols_marcados)
                newTeamInfo.ultimosGolsSofridos.append(teamInfoDataset.gols_sofridos)
                newTeamInfo.ultimosResultados.append(teamInfoDataset.is_winner)
                newTeamInfo.mediaUltimosGolsMarcados = sum(newTeamInfo.ultimosGolsFeitos) / len(newTeamInfo.ultimosGolsFeitos)
                newTeamInfo.mediaUltimosGolsSofridos = sum(newTeamInfo.ultimosGolsSofridos) / len(newTeamInfo.ultimosGolsSofridos)

        return newTeamInfo


    def obterUltimaInfoTeamArray(self, arrTeamInfoDataset: list[TeamInfoDataSet], id_team: int, id_season: int = None) -> TeamInfoDataSet:
        for teamInfo in list(reversed(arrTeamInfoDataset)):
            teamInfo: TeamInfoDataSet = teamInfo
            if id_season is not None:
                if teamInfo.id_team == id_team and teamInfo.id_season == id_season:
                    return teamInfo
            else:
                if teamInfo.id_team == id_team:
                    return teamInfo

        return None

    def obterUltimaTeamPlay(self, arrTeamsPlaysEntrada: list[TeamsPlaysEntrada], id_team: int, id_season: int = None) -> TeamsPlaysEntrada:
        for teamsPlay in list(reversed(arrTeamsPlaysEntrada)):
            teamsPlay: TeamsPlaysEntrada = teamsPlay
            if id_season is not None:
                if teamsPlay.id_season == id_season and (teamsPlay.id_team_home == id_team or teamsPlay.id_team_away == id_team):
                    return teamsPlay
            else:
                if teamsPlay.id_team_home == id_team or teamsPlay.id_team_away == id_team:
                    return teamsPlay

        return None

    def atualizarUltimaInfoTeamArray(self, arrTeamInfoDataset: list, id_team: int, name_atributo: str,
                                     value_atributo: int, id_season: int = None) -> TeamInfoDataSet:
        arrTeamInfoDataset.reverse()
        for teamInfo in arrTeamInfoDataset:
            if id_season is not None:
                if teamInfo.id_team == id_team and teamInfo.id_season == id_season:
                    setattr(teamInfo, name_atributo, value_atributo)
                    break
            else:
                if teamInfo.id_team == id_team:
                    setattr(teamInfo, name_atributo, value_atributo)
                    break
        arrTeamInfoDataset.reverse()


    def atualizarUltimaTeamsPlay(self, arrTeamsPlaysEntrada: list[TeamsPlaysEntrada], id_team: int, name_atributo: str,
                                     value_atributo: int, id_season: int = None) -> None:
        for teamsPlay in list(reversed(arrTeamsPlaysEntrada)):
            if id_season is not None:
                if teamsPlay.id_season == id_season and (teamsPlay.id_team_home == id_team or teamsPlay.id_team_away == id_team):
                    setattr(teamsPlay, name_atributo, value_atributo)
                    break
            else:
                if teamsPlay.id_team_home == id_team or teamsPlay.id_team_away == id_team:
                    setattr(teamsPlay, name_atributo, value_atributo)
                    break


    def calcularMediaGolsUltimosJogos(self, arrTeamInfoDataset: list[TeamInfoDataSet], id_team: int,
                                      id_season: int = None, nUltimosJogos: int = 6) -> list[int, int]:
        arrGolsMarcados = []
        arrGolsSofridos = []

        arrTeamInfoDataset.reverse()
        for teamInfo in arrTeamInfoDataset:
            if id_season is not None:
                if teamInfo.id_team == id_team and teamInfo.id_season == id_season:
                    arrGolsMarcados.append(teamInfo.gols_marcados)
                    arrGolsSofridos.append(teamInfo.gols_sofridos)
            else:
                if teamInfo.id_team == id_team:
                    arrGolsMarcados.append(teamInfo.gols_marcados)
                    arrGolsSofridos.append(teamInfo.gols_sofridos)

            if len(arrGolsMarcados) >= nUltimosJogos:
                break

        arrTeamInfoDataset.reverse()

        mediaGols = []

        if len(arrGolsMarcados) == 0:
            mediaGols.append(1)
        else:
            media = sum(arrGolsMarcados) / len(arrGolsMarcados)
            if media == 0:
                mediaGols.append(1)
            else:
                mediaGols.append(media)


        if len(arrGolsSofridos) == 0:
            mediaGols.append(1)
        else:
            media = sum(arrGolsSofridos) / len(arrGolsSofridos)

            if media == 0:
                mediaGols.append(1)
            else:
                mediaGols.append(media)

        return mediaGols


    def calcularMediaGolsTeamsPlay(self, arrTeamsPlaysEntrada: list[TeamsPlaysEntrada], id_team: int,
                                   id_season: int = None, nUltimosJogos: int = 6):
        arrGolsMarcados = []
        arrGolsSofridos = []

        for teamsPlay in list(reversed(arrTeamsPlaysEntrada)):
            if id_season is not None:
                if teamsPlay.id_team_home == id_team and teamsPlay.id_season == id_season:
                    arrGolsMarcados.append(teamsPlay.qtde_gols_marcados_home)
                    arrGolsSofridos.append(teamsPlay.qtde_gols_marcados_away)
                elif teamsPlay.id_team_away == id_team and teamsPlay.id_season == id_season:
                    arrGolsMarcados.append(teamsPlay.qtde_gols_marcados_away)
                    arrGolsSofridos.append(teamsPlay.qtde_gols_marcados_home)
            else:
                if teamsPlay.id_team_home == id_team:
                    arrGolsMarcados.append(teamsPlay.qtde_gols_marcados_home)
                    arrGolsSofridos.append(teamsPlay.qtde_gols_marcados_away)
                elif teamsPlay.id_team_away == id_team:
                    arrGolsMarcados.append(teamsPlay.qtde_gols_marcados_away)
                    arrGolsSofridos.append(teamsPlay.qtde_gols_marcados_home)

            if len(arrGolsMarcados) >= nUltimosJogos:
                break

        if len(arrGolsMarcados) == 0:
            return 0, 0
        else:
            mediaGolsMarcados = sum(arrGolsMarcados) / len(arrGolsMarcados)
            mediaGolsSofridos = sum(arrGolsSofridos) / len(arrGolsSofridos)

            return mediaGolsMarcados, mediaGolsSofridos


    def obterPontuacao(self, teamsPlays: TeamsPlaysEntrada, id_team: int) -> int:
        pontos = 0
        if teamsPlays.id_team_home == id_team:
            pontos = teamsPlays.qtde_pontos_season_home
            if teamsPlays.saida_prevista.is_winner_home == 1:
                pontos += 1
            elif teamsPlays.saida_prevista.is_winner_home == 2:
                pontos += 3

        elif teamsPlays.id_team_away == id_team:
            pontos = teamsPlays.qtde_pontos_season_away
            if teamsPlays.saida_prevista.is_winner_away == 1:
                pontos += 1
            elif teamsPlays.saida_prevista.is_winner_away == 2:
                pontos += 3

        return pontos

    def obterSaldoGols(self, teamsPlays: TeamsPlaysEntrada, id_team: int) -> int:
        saldoGols = 0
        if teamsPlays.id_team_home == id_team:
            saldoGols = teamsPlays.qtde_saldo_gols_home
            saldoGols += teamsPlays.qtde_gols_marcados_home
            saldoGols -= teamsPlays.qtde_gols_marcados_away

        elif teamsPlays.id_team_away == id_team:
            saldoGols = teamsPlays.qtde_saldo_gols_away
            saldoGols += teamsPlays.qtde_gols_marcados_away
            saldoGols -= teamsPlays.qtde_gols_marcados_home

        return saldoGols


