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
from api.regras.iaRNNRegras import DatasetRNN, CamadaSaidaRNN
from api.regras.iaRegras import IARegras

class TeamsPlaysSaidaPartida:
    def __init__(self):
        self.is_winner_home: int = None
        self.is_winner_away: int = None
        self.probabilidades_home: int = None
        self.probabilidades_away: int = None
        self.qtde_gols_marcados_home: int = None
        self.qtde_gols_marcados_away: int = None

class TeamsPlaysEntrada:
    def __init__(self):
        self.is_prever: int = None
        self.data_fixture: int = None
        self.id_season: int = None
        self.id_league: int = None
        #self.other_team_winner_opponent: int = None

        self.name_team_home: str = None
        self.id_country_team_home: int = None
        self.id_team_home: int = None
        self.qtde_pontos_season_home: int = None
        self.qtde_saldo_gols_home: int = None
        self.media_gols_marcados_home: int = None
        self.media_gols_sofridos_home: int = None
        self.qtde_gols_marcados_home: int = None

        self.media_vitorias_home: int = None
        self.is_decline_media_vitorias_home: int = None
        self.qtde_vitorias_home: int = None
        self.media_derrotas_home: int = None
        self.is_decline_media_derrotas_home: int = None
        self.qtde_derrotas_home: int = None
        self.media_empates_home: int = None
        self.is_decline_media_empates_home: int = None
        self.qtde_empates_home: int = None


        self.name_team_away: str = None
        self.id_country_team_away: int = None
        self.id_team_away: int = None
        self.qtde_pontos_season_away: int = None
        self.qtde_saldo_gols_away: int = None
        self.media_gols_marcados_away: int = None
        self.media_gols_sofridos_away: int = None
        self.qtde_gols_marcados_away: int = None

        self.media_vitorias_away: int = None
        self.is_decline_media_vitorias_away: int = None
        self.qtde_vitorias_away: int = None
        self.media_derrotas_away: int = None
        self.is_decline_media_derrotas_away: int = None
        self.qtde_derrotas_away: int = None
        self.media_empates_away: int = None
        self.is_decline_media_empates_away: int = None
        self.qtde_empates_away: int = None

        self.saida_prevista_partida: TeamsPlaysSaidaPartida = TeamsPlaysSaidaPartida()

    def get_value_atrribute(self, name_atribute: str) -> any:
        attr = self.__getattribute__(name_atribute)
        return attr

class StatisticsRegras:
    def __init__(self):
        self.iaRegras = IARegras()
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

    def obterAllFixturesByIdTeams(self, idTeamPrincipal: int, idTeamAdversario: int = None, id_season: int = None) -> list[TeamsPlaysEntrada]:
        arrKeysFinished = ['FT', 'AET', 'PEN', 'CANC']
        arrFixtures: list[Fixture] = self.fixturesRegras.obterTodasASFixturesSeasonAllTeamsByIdTeam(
            idTeamHome=idTeamPrincipal,
            idTeamAway=idTeamAdversario,
            id_season=id_season)

        arrTeamsPlays: list[TeamsPlaysEntrada] = []

        for fixture in arrFixtures:
            fixture.season: Season = fixture.season

            newTeamPlays = TeamsPlaysEntrada()
            newTeamPlays.id_season = fixture.id_season
            newTeamPlays.id_league = fixture.season.id_league
            newTeamPlays.data_fixture = fixture.date
            newTeamPlays.is_prever = 1 if fixture.status not in arrKeysFinished else 0
            newTeamPlays.saida_prevista_partida.qtde_gols_marcados_home = 0
            newTeamPlays.saida_prevista_partida.qtde_gols_marcados_away = 0

            fixture.teams: list[FixtureTeams] = fixture.teams
            indexOutherTeam = 1

            arrIdsHomeAway = [idTeamPrincipal, idTeamAdversario]
            isFixtureTreino = False
            if fixture.teams[0].id_team in arrIdsHomeAway or fixture.teams[1].id_team in arrIdsHomeAway:
                isFixtureTreino = True

            if newTeamPlays.is_prever == 1 and isFixtureTreino:
                print("Vai prever o jogo para a data: ", fixture.date)
                dateFutura = (datetime.now() + timedelta(days=2.0)).strftime("%Y-%m-%d")

                if fixture.date.strftime("%Y-%m-%d") < datetime.now().strftime("%Y-%m-%d"):
                    raise "Erro sem fixture"

            for team in fixture.teams:
                #if team.id_team == idTeamHome or (team.id_team != idTeamAway and fixture.teams[indexOutherTeam].id_team != idTeamHome):
                if team.is_home == 1:

                    newTeamPlays.id_team_home = team.id_team
                    dataTeamHome: Team = self.teamsRegras.teamsModel.obterByColumnsID(arrDados=[team.id_team])[0]
                    newTeamPlays.name_team_home = dataTeamHome.name
                    newTeamPlays.id_country_team_home = dataTeamHome.id_country

                    if team.is_winner == 0:
                        newTeamPlays.saida_prevista_partida.probabilidades_home = 0
                        newTeamPlays.saida_prevista_partida.is_winner_home = 0
                    elif team.is_winner is None:
                        newTeamPlays.saida_prevista_partida.is_winner_home = 0
                        newTeamPlays.saida_prevista_partida.probabilidades_home = 1
                    elif team.is_winner == 1:
                        newTeamPlays.saida_prevista_partida.is_winner_home = 1
                        newTeamPlays.saida_prevista_partida.probabilidades_home = 2

                    newTeamPlays.qtde_gols_marcados_home = team.goals
                    newTeamPlays.saida_prevista_partida.qtde_gols_marcados_home += team.goals

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

                    newTeamPlays.media_gols_marcados_home, newTeamPlays.media_gols_sofridos_home = \
                        self.calcularMediaGolsTeamsPlay(arrTeamsPlaysEntrada=arrTeamsPlays,
                                                        id_team=newTeamPlays.id_team_home,
                                                        id_season=newTeamPlays.id_season)

                    newTeamPlays.qtde_vitorias_home, newTeamPlays.media_vitorias_home, newTeamPlays.is_decline_media_vitorias_home = self.calcularMediaVDETeamsPlay(
                        arrTeamsPlaysEntrada=arrTeamsPlays, id_team=newTeamPlays.id_team_home, is_home=True,
                        id_season=None, typeInfo="V")

                    newTeamPlays.qtde_derrotas_home, newTeamPlays.media_derrotas_home, newTeamPlays.is_decline_media_derrotas_home = self.calcularMediaVDETeamsPlay(
                        arrTeamsPlaysEntrada=arrTeamsPlays, id_team=newTeamPlays.id_team_home, is_home=True,
                        id_season=None, typeInfo="D")

                    newTeamPlays.qtde_empates_home, newTeamPlays.media_empates_home, newTeamPlays.is_decline_media_empates_home = self.calcularMediaVDETeamsPlay(
                        arrTeamsPlaysEntrada=arrTeamsPlays, id_team=newTeamPlays.id_team_home, is_home=True,
                        id_season=None, typeInfo="E")

                else:
                    newTeamPlays.id_team_away = team.id_team
                    dataTeamAway: Team = self.teamsRegras.teamsModel.obterByColumnsID(arrDados=[team.id_team])[0]
                    newTeamPlays.name_team_away = dataTeamAway.name
                    newTeamPlays.id_country_team_away = dataTeamAway.id_country

                    ## Is winner está sendo sobreposto verificar URGENTE>
                    if team.is_winner == 0:
                        newTeamPlays.saida_prevista_partida.probabilidades_away = 0
                        newTeamPlays.saida_prevista_partida.is_winner_away = 0
                    elif team.is_winner is None:
                        newTeamPlays.saida_prevista_partida.probabilidades_away = 1
                        newTeamPlays.saida_prevista_partida.is_winner_away = 1
                    elif team.is_winner == 1:
                        newTeamPlays.saida_prevista_partida.probabilidades_away = 2
                        newTeamPlays.saida_prevista_partida.is_winner_away = 1


                    newTeamPlays.qtde_gols_marcados_away = team.goals
                    newTeamPlays.saida_prevista_partida.qtde_gols_marcados_away += team.goals

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

                    newTeamPlays.media_gols_marcados_away, newTeamPlays.media_gols_sofridos_away = \
                        self.calcularMediaGolsTeamsPlay(arrTeamsPlaysEntrada=arrTeamsPlays,
                                                        id_team=newTeamPlays.id_team_away,
                                                        id_season=newTeamPlays.id_season)

                    newTeamPlays.qtde_vitorias_away, newTeamPlays.media_vitorias_away, newTeamPlays.is_decline_media_vitorias_away = self.calcularMediaVDETeamsPlay(
                        arrTeamsPlaysEntrada=arrTeamsPlays, id_team=newTeamPlays.id_team_away, is_home=False,
                        id_season=None, typeInfo="V")

                    newTeamPlays.qtde_derrotas_away, newTeamPlays.media_derrotas_away, newTeamPlays.is_decline_media_derrotas_away = self.calcularMediaVDETeamsPlay(
                        arrTeamsPlaysEntrada=arrTeamsPlays, id_team=newTeamPlays.id_team_away, is_home=False,
                        id_season=None, typeInfo="D")

                    newTeamPlays.qtde_empates_away, newTeamPlays.media_empates_away, newTeamPlays.is_decline_media_empates_away = self.calcularMediaVDETeamsPlay(
                        arrTeamsPlaysEntrada=arrTeamsPlays, id_team=newTeamPlays.id_team_away, is_home=False,
                        id_season=None, typeInfo="E")

                indexOutherTeam = 0

            arrTeamsPlays.append(newTeamPlays)
        return arrTeamsPlays


    def normalizarDadosTeamsPlayDataset(self, arrTeamsPlays: list[TeamsPlaysEntrada], arrIdsTeamPrever: list[int],
                                        qtdeDados: int, isFiltrarTeams: bool = False, isPartida: bool = True) -> DatasetRNN:
        arrKeysIgnorarDadosEntrada: list = ["data_fixture", "is_prever", "name_team_home", "name_team_away",
                                            "saida_prevista_partida", "qtde_gols_marcados_away",
                                            "qtde_gols_marcados_home", "qtde_gols_marcados"]

        arrKeysIgnorarDadosEsperados: list = ["qtde_gols_marcados", "qtde_gols_marcados_away", "qtde_gols_marcados_home"]

        if isPartida:
            arrKeysIgnorarDadosEsperados.append("is_winner_away")
            arrKeysIgnorarDadosEsperados.append("is_winner_home")
        else:
            arrKeysIgnorarDadosEsperados.append("probabilidades_away")
            arrKeysIgnorarDadosEsperados.append("probabilidades_home")

        arrDadosEntrada: list = []
        arrDadosEsperadosPartida: list = []
        arrDadosPrever: list = []

        ordemNameValuesEntrada: list[str] = []
        ordemNameValuesSaidaPartida: list[str] = []

        isSalvouOrdem = False

        qtdeDadosTeamPrincipal = 0
        qtdeDadosTeamAdversario = 0
        qtdeDadosLevadosEmConsideracao = 0

        for team in list(reversed(arrTeamsPlays)):
            qtdeDadosLevadosEmConsideracao += 1
            if isFiltrarTeams:
                if team.id_team_away not in arrIdsTeamPrever and team.id_team_home not in arrIdsTeamPrever:
                    continue

                if len(arrIdsTeamPrever) == 1:
                    if (team.id_team_home == arrIdsTeamPrever[0] or team.id_team_away == arrIdsTeamPrever[0]):
                        if qtdeDadosTeamPrincipal <= qtdeDados:
                            qtdeDadosTeamPrincipal += 1

                        if qtdeDadosTeamPrincipal > qtdeDados:
                            continue

                if len(arrIdsTeamPrever) == 2:
                    if (team.id_team_home == arrIdsTeamPrever[0] or team.id_team_away == arrIdsTeamPrever[0]):
                        if qtdeDadosTeamPrincipal <= qtdeDados:
                            qtdeDadosTeamPrincipal += 1

                        if qtdeDadosTeamPrincipal > qtdeDados:
                            continue

                    if (team.id_team_home == arrIdsTeamPrever[1] or team.id_team_away == arrIdsTeamPrever[1]):
                        if qtdeDadosTeamAdversario <= qtdeDados:
                            qtdeDadosTeamAdversario += 1

                        if qtdeDadosTeamAdversario > qtdeDados:
                            continue

            teamEntradaDict = team.__dict__
            teamSaidaPartidaDict = team.saida_prevista_partida.__dict__


            newDadosEntradas = []
            newDadosEsperadosPartida = []

            for key in teamEntradaDict.keys():
                if key in arrKeysIgnorarDadosEntrada:
                    continue
                else:
                    if not isSalvouOrdem:
                        ordemNameValuesEntrada.append(key)
                    newDadosEntradas.append(teamEntradaDict[key])

            for key in teamSaidaPartidaDict:
                if key in arrKeysIgnorarDadosEsperados:
                    continue
                else:
                    if not isSalvouOrdem:
                        ordemNameValuesSaidaPartida.append(key)
                    newDadosEsperadosPartida.append(teamSaidaPartidaDict[key])

            isSalvouOrdem = True

            if team.is_prever == 1:
                arrDadosPrever.insert(0, newDadosEntradas)
            else:
                arrDadosEntrada.insert(0, newDadosEntradas)
                arrDadosEsperadosPartida.insert(0, newDadosEsperadosPartida)


        arrDadosEntradaNormalizados, max_valor, min_valor = self.iaRegras.normalizar_dataset(dataset=arrDadosEntrada)
        arrDadosPreverNormalizados = self.iaRegras.normalizar_dataset(dataset=arrDadosPrever, max_valor=max_valor, min_valor=min_valor)[0]
        dados_normalizados, max_value_esperado_partida, min_value_esperado_partida = self.iaRegras.normalizar_dataset(dataset=arrDadosEsperadosPartida)

        arrCamadasSaidasPartida = [CamadaSaidaRNN("softmax", [], maxValue + 1) for maxValue in max_value_esperado_partida]
        for dadoEsperadosPartida in arrDadosEsperadosPartida:
            for indexDadoEsperados in range(len(dadoEsperadosPartida)):
                camadasSaidasPartida = numpy.zeros(max_value_esperado_partida[indexDadoEsperados] + 1, dtype=numpy.int32)
                camadasSaidasPartida[dadoEsperadosPartida[indexDadoEsperados]] = 1
                arrCamadasSaidasPartida[indexDadoEsperados].arr_saidas_esperas.append(camadasSaidasPartida)

        newDatasetNormalizado = DatasetRNN()
        newDatasetNormalizado.arr_entradas_treino = arrDadosEntradaNormalizados
        newDatasetNormalizado.arr_prevevisao = arrDadosPreverNormalizados
        newDatasetNormalizado.max_value_entradas = list(max_valor)
        newDatasetNormalizado.min_value_entradas = list(min_valor)
        newDatasetNormalizado.max_value_esperados = list(max_value_esperado_partida)
        newDatasetNormalizado.min_value_esperados = list(min_value_esperado_partida)
        newDatasetNormalizado.arr_name_values_entrada = ordemNameValuesEntrada
        newDatasetNormalizado.dado_exemplo = arrDadosPrever
        newDatasetNormalizado.quantia_dados = len(arrDadosEntradaNormalizados)
        newDatasetNormalizado.quantia_neuronios_entrada = len(arrDadosEntradaNormalizados[0])

        newDatasetNormalizado.arr_camadas_saidas = arrCamadasSaidasPartida
        newDatasetNormalizado.arr_name_values_saida = ordemNameValuesSaidaPartida


        return newDatasetNormalizado, qtdeDadosLevadosEmConsideracao, qtdeDadosTeamPrincipal, qtdeDadosTeamAdversario


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
            if teamsPlays.saida_prevista_partida.probabilidades_home == 1:
                pontos += 1
            elif teamsPlays.saida_prevista_partida.probabilidades_home == 2:
                pontos += 3

        elif teamsPlays.id_team_away == id_team:
            pontos = teamsPlays.qtde_pontos_season_away
            if teamsPlays.saida_prevista_partida.probabilidades_away == 1:
                pontos += 1
            elif teamsPlays.saida_prevista_partida.probabilidades_away == 2:
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

    def calcularMediaVDETeamsPlay(self, arrTeamsPlaysEntrada: list[TeamsPlaysEntrada], id_team: int, is_home: bool,
                                   id_season: int = None, nUltimosJogos: int = 6, typeInfo: str = None) -> list[int, int]:
        """
            type info qtde: V, D ou E
        """
        arrUltimasVitorias = []
        arrUltimasDerotas = []
        arrUltimosEmpates = []
        keyChave = "_home" if is_home else "_away"

        for teamsPlay in list(reversed(arrTeamsPlaysEntrada)):
            if is_home:
                val_winner = teamsPlay.saida_prevista_partida.probabilidades_home
            else:
                val_winner = teamsPlay.saida_prevista_partida.probabilidades_away

            if id_season is not None:
                if (teamsPlay.id_team_home == id_team or teamsPlay.id_team_away == id_team) \
                        and teamsPlay.id_season == id_season:
                    if val_winner == 0:
                        arrUltimasDerotas.append(1)
                        arrUltimasVitorias.append(0)
                        arrUltimosEmpates.append(0)
                    elif val_winner == 1:
                        arrUltimosEmpates.append(1)
                        arrUltimasVitorias.append(0)
                        arrUltimasDerotas.append(0)
                    elif val_winner == 2:
                        arrUltimasVitorias.append(1)
                        arrUltimosEmpates.append(0)
                        arrUltimasDerotas.append(0)
            else:
                if teamsPlay.id_team_home == id_team or teamsPlay.id_team_away == id_team:
                    if val_winner == 0:
                        arrUltimasDerotas.append(1)
                        arrUltimasVitorias.append(0)
                        arrUltimosEmpates.append(0)
                    elif val_winner == 1:
                        arrUltimosEmpates.append(1)
                        arrUltimasVitorias.append(0)
                        arrUltimasDerotas.append(0)
                    elif val_winner == 2:
                        arrUltimasVitorias.append(1)
                        arrUltimosEmpates.append(0)
                        arrUltimasDerotas.append(0)

            if (len(arrUltimasVitorias) + len(arrUltimosEmpates) + len(arrUltimasDerotas)) / 3 >= nUltimosJogos:
                break

        if typeInfo == "V":
            qtdeVitorias = 0 if len(arrUltimasVitorias) == 0 else sum(arrUltimasVitorias)
            mediaVitorias = 0 if len(arrUltimasVitorias) == 0 else sum(arrUltimasVitorias) / len(arrUltimasVitorias)
            attr_name = "media_vitorias" + keyChave
            isDeclineMediaVitorias = 0 if len(arrTeamsPlaysEntrada) == 0 else int(arrTeamsPlaysEntrada[-1].get_value_atrribute(name_atribute=attr_name) < mediaVitorias)
            return qtdeVitorias, mediaVitorias, isDeclineMediaVitorias
        elif typeInfo == "D":
            qtdeDerrotas = 0 if len(arrUltimasDerotas) == 0 else sum(arrUltimasDerotas)
            mediaDerrotas = 0 if len(arrUltimasDerotas) == 0 else sum(arrUltimasDerotas) / len(arrUltimasDerotas)
            attr_name = "media_derrotas" + keyChave
            isDeclineMediaDerrotas = 0 if len(arrTeamsPlaysEntrada) == 0 else int(arrTeamsPlaysEntrada[-1].get_value_atrribute(name_atribute=attr_name) < mediaDerrotas)
            return qtdeDerrotas, mediaDerrotas, isDeclineMediaDerrotas
        elif typeInfo == "E":
            qtdeEmpates = 0 if len(arrUltimosEmpates) == 0 else sum(arrUltimosEmpates)
            mediaEmpates = 0 if len(arrUltimosEmpates) == 0 else sum(arrUltimosEmpates) / len(arrUltimosEmpates)
            attr_name = "media_empates" + keyChave
            isDeclineMediaEmpates = 0 if len(arrTeamsPlaysEntrada) == 0 else int(arrTeamsPlaysEntrada[-1].get_value_atrribute(name_atribute=attr_name) < mediaEmpates)
            return qtdeEmpates, mediaEmpates, isDeclineMediaEmpates


    def last_opponent_winner(self, id_home_or_away: int, id_opponent: int, data_fixture: str = None):
        arrFixtures: list[Fixture] = self.fixturesRegras.obter(id_team=id_home_or_away)
        isWinner: int = None
        for fixture in list(reversed(arrFixtures)):
            fixture.teams: list[FixtureTeams] = fixture.teams

            if fixture.teams[0].id_team == id_opponent or fixture.teams[1].id_team == id_opponent:
                if fixture.teams[0].id_team == id_home_or_away:
                    isWinner = 1 if fixture.teams[0].is_winner is None else 2 if fixture.teams[0].is_winner == 1 else 0
                elif fixture.teams[1].id_team == id_home_or_away:
                    isWinner = 1 if fixture.teams[1].is_winner is None else 2 if fixture.teams[1].is_winner == 1 else 0
                break

        if isWinner is None:
            return -1
        else:
            return isWinner