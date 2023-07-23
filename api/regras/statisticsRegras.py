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
from api.regras.iaUteisRegras import IAUteisRegras

class TeamsPlaysSaidaPartida:
    def __init__(self):
        self.probabilidades_home: int = None
        self.probabilidades_away: int = None
        self.probabilidades_partida: int = None
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
        self.is_playing_home_home: str = None
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
        self.is_playing_home_away: str = None
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

class TeamsPlaysDataset:
    def __init__(self):
        self.data_previsao: str = None
        self.arr_name_coluns_entrada: list[str] = []
        self.max_value_entrada: list = []
        self.min_value_entrada: list = []
        self.arr_dados_entrada_original: list[list] = []
        self.arr_dados_entrada: list[list] = []

        self.arr_dados_prever_original: list[list] = []
        self.arr_dados_prever: list[list] = []

        self.arr_name_coluns_rotulos: list[str] = []
        self.max_value_rotulos: list = []
        self.min_value_rotulos: list = []
        self.arr_dados_rotulos_original: list[list[list]] = []
        self.arr_dados_rotulos_in_camadas: list[list[list]] = []
        self.arr_dados_rotulos_in_camadas_normalizados: list[list[list]] = []

class StatisticsRegras:
    def __init__(self):
        self.iaRegras = IAUteisRegras()
        self.teamsRegras = TeamsRegras()
        self.seasonRegras = SeasonsRegras()
        self.fixturesRegras = FixturesRegras()

    def obterAllFixturesByIdTeams(self, idTeamPrincipal: int, idTeamAdversario: int = None, id_season: int = None,
                                  isFiltrarTeams: bool = True, qtdeDados: int = 40,
                                  qtdeJogosAnterioresPrever: int = None) -> list[TeamsPlaysEntrada]:
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

            arrIdsHomeAway = [idTeamPrincipal]
            if idTeamAdversario is not None:
                arrIdsHomeAway.append(idTeamAdversario)

            isFixtureTreino = False
            if fixture.teams[0].id_team in arrIdsHomeAway or fixture.teams[1].id_team in arrIdsHomeAway:
                isFixtureTreino = True

            if newTeamPlays.is_prever == 1 and isFixtureTreino:
                print("Vai prever o jogo para a data: ", fixture.date)
                dateFutura = (datetime.now() + timedelta(days=2.0)).strftime("%Y-%m-%d")

                if fixture.date.strftime("%Y-%m-%d") < datetime.now().strftime("%Y-%m-%d"):
                    #raise "Erro sem fixture"
                    pass

            for team in fixture.teams:
                #if team.id_team == idTeamHome or (team.id_team != idTeamAway and fixture.teams[indexOutherTeam].id_team != idTeamHome):
                if team.is_home == 1:
                    newTeamPlays.is_playing_home_home = 1
                    newTeamPlays.id_team_home = team.id_team
                    dataTeamHome: Team = self.teamsRegras.teamsModel.obterByColumnsID(arrDados=[team.id_team])[0]
                    newTeamPlays.name_team_home = dataTeamHome.name
                    newTeamPlays.id_country_team_home = dataTeamHome.id_country

                    if team.is_winner == 0:
                        newTeamPlays.saida_prevista_partida.probabilidades_home = 0
                        newTeamPlays.saida_prevista_partida.probabilidades_partida = 2
                    elif team.is_winner is None:
                        newTeamPlays.saida_prevista_partida.probabilidades_home = 1
                        newTeamPlays.saida_prevista_partida.probabilidades_partida = 1
                    elif team.is_winner == 1:
                        newTeamPlays.saida_prevista_partida.probabilidades_home = 2
                        newTeamPlays.saida_prevista_partida.probabilidades_partida = 0

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
                    newTeamPlays.is_playing_home_away = 0
                    dataTeamAway: Team = self.teamsRegras.teamsModel.obterByColumnsID(arrDados=[team.id_team])[0]
                    newTeamPlays.name_team_away = dataTeamAway.name
                    newTeamPlays.id_country_team_away = dataTeamAway.id_country

                    ## Is winner está sendo sobreposto verificar URGENTE>
                    if team.is_winner == 0:
                        newTeamPlays.saida_prevista_partida.probabilidades_away = 0
                        newTeamPlays.saida_prevista_partida.probabilidades_partida = 0
                    elif team.is_winner is None:
                        newTeamPlays.saida_prevista_partida.probabilidades_away = 1
                        newTeamPlays.saida_prevista_partida.probabilidades_partida = 1
                    elif team.is_winner == 1:
                        newTeamPlays.saida_prevista_partida.probabilidades_away = 2
                        newTeamPlays.saida_prevista_partida.probabilidades_partida = 2

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

        arrTeamsPlaysFiltrado = []
        qtdeDadosTeamPrincipal = 0
        qtdeDadosTeamAdversario = 0

        if isFiltrarTeams:
            for team in list(reversed(arrTeamsPlays)):
                if team.id_team_away not in arrIdsHomeAway and team.id_team_home not in arrIdsHomeAway:
                    continue

                if (team.id_team_home == arrIdsHomeAway[0] or team.id_team_away == arrIdsHomeAway[0]) and \
                        qtdeDadosTeamPrincipal < qtdeDados:
                    qtdeDadosTeamPrincipal += 1
                    arrTeamsPlaysFiltrado.insert(0, team)


                elif len(arrIdsHomeAway) == 2:
                    if (team.id_team_home == arrIdsHomeAway[1] or team.id_team_away == arrIdsHomeAway[1]) and \
                            qtdeDadosTeamAdversario < qtdeDados:
                        qtdeDadosTeamAdversario += 1
                        arrTeamsPlaysFiltrado.insert(0, team)
        else:
            arrTeamsPlaysFiltrado = arrTeamsPlays


        if qtdeJogosAnterioresPrever is not None:
            for indexTeam in range(len(arrTeamsPlaysFiltrado)):
                if arrTeamsPlaysFiltrado[indexTeam].is_prever == 1:
                    continue
                else:
                    if indexTeam <= qtdeJogosAnterioresPrever:
                        arrTeamsPlaysFiltrado[indexTeam].is_prever = 1

        return arrTeamsPlaysFiltrado


    def normalizarDadosTeamsPlayDataset(self, arrTeamsPlays: list[TeamsPlaysEntrada],
                                        arrIdsTeamPrever: list[int]) -> TeamsPlaysDataset:

        arrKeysIgnorarDadosEntrada: list = ["data_fixture", "is_prever", "name_team_home", "name_team_away",
                                            "saida_prevista_partida", "qtde_gols_marcados_away",
                                            "qtde_gols_marcados_home", "qtde_gols_marcados", "id_country_team_home",
                                            "id_country_team_away", "id_league", "id_team_home", "id_team_away"]

        arrKeysIgnorarDadosEsperados: list = ["qtde_gols_marcados", "qtde_gols_marcados_away",
                                              "qtde_gols_marcados_home", "probabilidades_home", "probabilidades_away"]

        arrDadosEntrada: list = []
        arrDadosEsperadosPartida: list = []
        arrDadosPrever: list = []

        ordemNameValuesEntrada: list[str] = []
        ordemNameValuesSaidaPartida: list[str] = []

        isSalvouOrdem = False
        data_previsao = None

        for team in arrTeamsPlays:
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
                data_previsao = (team.data_fixture - timedelta(hours=3.0)).strftime("%Y-%m-%d %H:%M:%S")
                arrDadosPrever.append(newDadosEntradas)
            else:
                arrDadosEntrada.append(newDadosEntradas)
                arrDadosEsperadosPartida.append(newDadosEsperadosPartida)


        arrDadosEntradaNormalizados, max_valor, min_valor = self.iaRegras.normalizar_dataset(dataset=arrDadosEntrada)
        arrDadosPreverNormalizados = self.iaRegras.normalizar_dataset(dataset=arrDadosPrever, max_valor=max_valor, min_valor=min_valor)[0]
        arrDadosRotulosNormalizados, max_value_esperado_partida, min_value_esperado_partida = self.iaRegras.normalizar_dataset(dataset=arrDadosEsperadosPartida)

        arrDadosRotulosInCamadas = []
        for rotulo in arrDadosEsperadosPartida:
            camadas_saida = []
            for index_val_rotulo in range(len(rotulo)):
                camada_saida = [rotulo[index_val_rotulo]]
                camadas_saida.append(camada_saida)
            arrDadosRotulosInCamadas.append(camadas_saida)

        arrDadosRotulosInCamadasNormalizado = []
        for rotulo in arrDadosRotulosNormalizados:
            camadas_saida = []
            for index_val_rotulo in range(len(rotulo)):
                camada_saida = [rotulo[index_val_rotulo]]
                camadas_saida.append(camada_saida)
            arrDadosRotulosInCamadasNormalizado.append(camadas_saida)

        newTeamsPlaysDataset  = TeamsPlaysDataset()
        newTeamsPlaysDataset.data_previsao = data_previsao
        newTeamsPlaysDataset.arr_name_coluns_entrada = ordemNameValuesEntrada
        newTeamsPlaysDataset.arr_dados_entrada_original = arrDadosEntrada
        newTeamsPlaysDataset.arr_dados_entrada = arrDadosEntradaNormalizados
        newTeamsPlaysDataset.max_value_entrada = max_valor
        newTeamsPlaysDataset.min_value_entrada = min_valor
        newTeamsPlaysDataset.arr_dados_prever_original = arrDadosPrever
        newTeamsPlaysDataset.arr_dados_prever = arrDadosPreverNormalizados
        newTeamsPlaysDataset.arr_dados_rotulos_original = arrDadosEsperadosPartida
        newTeamsPlaysDataset.arr_name_coluns_rotulos = ordemNameValuesSaidaPartida
        newTeamsPlaysDataset.arr_dados_rotulos_in_camadas = arrDadosRotulosInCamadas
        newTeamsPlaysDataset.arr_dados_rotulos_in_camadas_normalizados = arrDadosRotulosInCamadasNormalizado
        newTeamsPlaysDataset.max_value_rotulos = max_value_esperado_partida
        newTeamsPlaysDataset.min_value_rotulos = min_value_esperado_partida

        return newTeamsPlaysDataset

    def obterDatasetNormalizadoUnicoPlaysTeam(self, arrTeamsPlays: list[TeamsPlaysEntrada], idTeam: int) -> TeamsPlaysDataset:
        arrNameDadosEntrada: list[str] = []
        arrNameDadosEsperados: list[str] = []
        arrDadosEntrada: list = []
        arrDadosEsperados: list = []
        arrDadosPrever: list = []
        isSalvouNomesEntrada = False
        isSalvouNomesSaida = False
        data_previsao = None

        for team in arrTeamsPlays:
            arrNameDadosEntrada = []
            arrNameDadosEsperados = []
            teamDict = team.__dict__
            saidaDict = team.saida_prevista_partida.__dict__
            arrDadosEntradaTeam = []
            arrDadosEsperadosTeam = []
            arrDadosPreverTeam = []
            keyFilterTeam = None
            keyFilterOutr = None

            if team.id_team_home == idTeam:
                keyFilterTeam = "_home"
                keyFilterOutr = "_away"
            else:
                keyFilterTeam = "_away"
                keyFilterOutr = "_home"

            arrKeysComunsGetData: list[str] = ["id_season"]
            arrKeysAbrevGetData: list[str] = ["is_playing_home", "id_team", "qtde_pontos_season", "qtde_saldo_gols",
                                              "media_gols_marcados", "media_gols_sofridos", "qtde_gols_marcados",
                                              "media_vitorias", "is_decline_media_vitorias", "qtde_vitorias",
                                              "media_derrotas", "is_decline_media_derrotas", "qtde_derrotas",
                                              "media_empates", "is_decline_media_empates", "qtde_empates"]

            arrKeysIgnorGetData: list[str] = ["data_fixture", "is_prever", "name_team", "saida_prevista_partida",
                                              "qtde_gols_marcados", "id_country_team_home", "id_league", "id_country_team_away"]

            arrKeysAbreSaida: list[str] = ["probabilidades"]

            keysEntradaDict = teamDict.keys()
            keysSaidaDict = saidaDict.keys()

            for keyComuns in arrKeysComunsGetData:
                for key in keysEntradaDict:
                    if keyComuns == key:
                        if team.is_prever == 1:
                            arrDadosPreverTeam.append(teamDict[keyComuns])
                        elif team.is_prever == 0:
                            arrDadosEntradaTeam.append(teamDict[keyComuns])

                        if not isSalvouNomesEntrada:
                            arrNameDadosEntrada.append(keyComuns)

            for keyAbrev in arrKeysAbrevGetData:
                for key in keysEntradaDict:
                    nameKey = keyAbrev + keyFilterTeam
                    if nameKey == key:
                        if team.is_prever == 1:
                            arrDadosPreverTeam.append(teamDict[nameKey])
                        elif team.is_prever == 0:
                            arrDadosEntradaTeam.append(teamDict[nameKey])

                        if not isSalvouNomesEntrada:
                            arrNameDadosEntrada.append(nameKey)

            for keyAbrev in arrKeysAbrevGetData:
                for key in keysEntradaDict:
                    nameKey = keyAbrev + keyFilterOutr
                    if nameKey == key:
                        if team.is_prever == 1:
                            arrDadosPreverTeam.append(teamDict[nameKey])
                        elif team.is_prever == 0:
                            arrDadosEntradaTeam.append(teamDict[nameKey])

                        if not isSalvouNomesEntrada:
                            arrNameDadosEntrada.append(nameKey)

            for keyAbrev in arrKeysAbreSaida:
                for key in keysSaidaDict:
                    nameKey = keyAbrev + keyFilterTeam
                    if nameKey == key:
                        if team.is_prever == 0:
                            arrDadosEsperadosTeam.append(saidaDict[nameKey])

                        if not isSalvouNomesSaida:
                            arrNameDadosEsperados.append(nameKey)

            if team.is_prever == 1:
                arrDadosPrever.append(arrDadosPreverTeam)
            elif team.is_prever == 0:
                arrDadosEntrada.append(arrDadosEntradaTeam)
                arrDadosEsperados.append(arrDadosEsperadosTeam)

            if team.is_prever == 1:
                data_previsao = (team.data_fixture - timedelta(hours=3.0)).strftime("%Y-%m-%d %H:%M:%S")
                isSalvouNomesEntrada = True

        arrDadosEntradaNormalizados, max_valor, min_valor = self.iaRegras.normalizar_dataset(dataset=arrDadosEntrada)
        arrDadosPreverNormalizados = self.iaRegras.normalizar_dataset(dataset=arrDadosPrever, max_valor=max_valor, min_valor=min_valor)[0]
        arrDadosRotulosNormalizados, max_value_esperado_partida, min_value_esperado_partida = self.iaRegras.normalizar_dataset(dataset=arrDadosEsperados)

        arrDadosRotulosInCamadas = []
        for rotulo in arrDadosEsperados:
            camadas_saida = []
            for index_val_rotulo in range(len(rotulo)):
                camada_saida = [rotulo[index_val_rotulo]]
                camadas_saida.append(camada_saida)
            arrDadosRotulosInCamadas.append(camadas_saida)

        arrDadosRotulosInCamadasNormalizado = []
        for rotulo in arrDadosRotulosNormalizados:
            camadas_saida = []
            for index_val_rotulo in range(len(rotulo)):
                camada_saida = [rotulo[index_val_rotulo]]
                camadas_saida.append(camada_saida)
            arrDadosRotulosInCamadasNormalizado.append(camadas_saida)

        newTeamPlaysDataset = TeamsPlaysDataset()
        newTeamPlaysDataset.data_previsao = data_previsao
        newTeamPlaysDataset.arr_name_coluns_entrada = arrNameDadosEntrada
        newTeamPlaysDataset.arr_dados_entrada_original = arrDadosEntrada
        newTeamPlaysDataset.arr_dados_prever_original = arrDadosPrever
        newTeamPlaysDataset.max_value_entrada = max_valor
        newTeamPlaysDataset.min_value_entrada = min_valor
        newTeamPlaysDataset.arr_name_coluns_rotulos = arrNameDadosEsperados
        newTeamPlaysDataset.arr_dados_rotulos_original = arrDadosEsperados
        newTeamPlaysDataset.max_value_rotulos = max_value_esperado_partida
        newTeamPlaysDataset.min_value_rotulos = min_value_esperado_partida

        newTeamPlaysDataset.arr_dados_entrada = arrDadosEntradaNormalizados
        newTeamPlaysDataset.arr_dados_prever = arrDadosPreverNormalizados
        newTeamPlaysDataset.arr_dados_rotulos_in_camadas = arrDadosRotulosInCamadas
        newTeamPlaysDataset.arr_dados_rotulos_in_camadas_normalizados = arrDadosRotulosInCamadasNormalizado

        return newTeamPlaysDataset

    def obterDatasetNormalizadoTeamsPlays(self, id_team_home: int = None, id_team_away: int = None, id_season: int = None,
                                          qtdeDados=40, isFiltrarTeams=True, isPartida=False) -> TeamsPlaysDataset:
        if id_team_home is None and id_team_away is None:
            raise "Passar pelo menos id_team_home ou id_team_away"

        arrTeamsPlays = self.obterAllFixturesByIdTeams(idTeamPrincipal=id_team_home, idTeamAdversario=id_team_away,
                                                       id_season=id_season, isFiltrarTeams=isFiltrarTeams, qtdeDados=qtdeDados)

        if isPartida:
            arrIdsTeam = []
            arrIdsTeam.append(id_team_home) if id_team_home is not None else None
            arrIdsTeam.append(id_team_away) if id_team_away is not None else None
            teamsPlaysDataset = self.normalizarDadosTeamsPlayDataset(arrTeamsPlays=arrTeamsPlays,
                                                                     arrIdsTeamPrever=arrIdsTeam)
        else:
            if id_team_home is not None and id_team_away is not None:
                raise "Passar somente id_team_home ou id_team_away"

            id_team = id_team_home if id_team_home is not None else id_team_away
            teamsPlaysDataset = self.obterDatasetNormalizadoUnicoPlaysTeam(arrTeamsPlays=arrTeamsPlays, idTeam=id_team)

        return teamsPlaysDataset

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
                                   id_season: int = None, nUltimosJogos: int = 5, typeInfo: str = None) -> list[int, int]:
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