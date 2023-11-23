import numpy

from typing import List
from enum import Enum
from copy import deepcopy
from scipy.stats import zscore

from api.models.fixturesModel import Fixture, FixturesModel
from api.models.fixturesTeamsModel import FixtureTeams
from api.models.fixturesTeamsStatisticsModel import FixtureTeamStatistic
from api.models.teamsModel import Team, TeamsModel


class EnumTypeStatistics(Enum):
    shots_on_goal: int = 1
    shots_off_goal: int = 2
    shots_total: int = 3
    shots_bloqued: int = 4
    fouls: int = 7
    corner_kicks: int = 8
    offsides: int = 9
    ball_possession: float = 10
    cards_yellow: int = 11
    cards_red: int = 12
    goalkeeper_saves: int = 13
    passes_total: int = 14
    passes_accurate: int = 15
    passes_precision: float = 16


class InfoTeamFixture:
    def __init__(self):
        self.lastInfoTeamFixture: InfoTeamFixture = None
        # Dados obtidos atraves do database

        # Infos da tabela fixture, Teams e Season
        self.date_fixture: str = None
        self.id_season: int = None
        self.id_fixture: int = None
        self.status_fixture: str = None
        self.time_elapsed: str = None
        self.is_statistics_fixture: int = None
        self.is_terminou_fulltime: int = None
        self.is_terminou_after_fulltime: int = None
        self.id_team: int = None
        self.name_team: str = None

        # Infos da tabela fixture_teams
        self.winner: int = None  # Pode ser 0 - Derrota, 1 - Empate, 2 - Vitoria
        self.is_home: int = None
        self.is_vitoria: int = None
        self.is_empate: int = None
        self.is_derrota: int = None

        self.gols_primeiro_tempo: int = None
        self.gols_primeiro_tempo_conceded: int = None
        self.gols_segundo_tempo: int = None
        self.gols_segundo_tempo_conceded: int = None
        self.goals_total: int = None
        self.goals_halftime: int = None
        self.goals_fulltime: int = None
        self.goals_extratime: int = None
        self.goals_penalty: int = None

        self.goals_total_conceded: int = None
        self.goals_halftime_conceded: int = None
        self.goals_fulltime_conceded: int = None
        self.goals_extratime_conceded: int = None
        self.goals_penalty_conceded: int = None

        # Infos da tabela fixture_teams_statistics
        self.shots_on_goal: int = None
        self.shots_off_goal: int = None
        self.shots_total: int = None
        self.shots_bloqued: int = None
        self.fouls: int = None
        self.corner_kicks: int = None
        self.offsides: int = None
        self.ball_possession: float = None
        self.cards_yellow: int = None
        self.cards_red: int = None
        self.goalkeeper_saves: int = None
        self.passes_total: int = None
        self.passes_accurate: int = None
        self.passes_precision: float = None

        # Dados obtidos através do calculos das InfoPartidaTeam anteriores
        self.pontos_season: int = None
        self.pontos_team: int = None
        self.saldo_goals_season: int = None
        self.saldo_goals_team: int = None

        self.media_gols_primeiro_tempo: int = None
        self.media_gols_primeiro_tempo_conceded: int = None
        self.media_gols_segundo_tempo: int = None
        self.media_gols_segundo_tempo_conceded: int = None
        self.media_goals_total: float = None
        self.media_goals_total_conceded: float = None
        self.media_goals_halftime: float = None
        self.media_goals_halftime_conceded: float = None
        self.media_goals_fulltime: float = None
        self.media_goals_fulltime_conceded: float = None

        self.media_gols_primeiro_tempo_home: int = None
        self.media_gols_primeiro_tempo_conceded_home: int = None
        self.media_gols_segundo_tempo_home: int = None
        self.media_gols_segundo_tempo_conceded_home: int = None
        self.media_goals_total_home: float = None
        self.media_goals_total_conceded_home: float = None
        self.media_goals_halftime_home: float = None
        self.media_goals_halftime_conceded_home: float = None
        self.media_goals_fulltime_home: float = None
        self.media_goals_fulltime_conceded_home: float = None

        self.media_gols_primeiro_tempo_away: int = None
        self.media_gols_primeiro_tempo_conceded_away: int = None
        self.media_gols_segundo_tempo_away: int = None
        self.media_gols_segundo_tempo_conceded_away: int = None
        self.media_goals_total_away: float = None
        self.media_goals_total_conceded_away: float = None
        self.media_goals_halftime_away: float = None
        self.media_goals_halftime_conceded_away: float = None
        self.media_goals_fulltime_away: float = None
        self.media_goals_fulltime_conceded_away: float = None

        self.media_vitorias: float = None
        self.media_empates: float = None
        self.media_derrotas: float = None

        self.media_vitorias_home: float = None
        self.media_empates_home: float = None
        self.media_derrotas_home: float = None

        self.media_vitorias_away: float = None
        self.media_empates_away: float = None
        self.media_derrotas_away: float = None

        self.media_shots_on_goal: float = None
        self.media_shots_off_goal: float = None
        self.media_shots_total: float = None
        self.media_shots_bloqued: float = None
        self.media_fouls: float = None
        self.media_corner_kicks: float = None
        self.media_offsides: float = None
        self.media_ball_possession: float = None
        self.media_cards_yellow: float = None
        self.media_cards_red: float = None
        self.media_goalkeeper_saves: float = None
        self.media_passes_total: float = None
        self.media_passes_accurate: float = None
        self.media_passes_precision: float = None


class DataseIteamtRotulo:
    pass


class DatasetItemEntrada:
    def __init__(self, currentInfoFixture: InfoTeamFixture, lastInfoFixture: InfoTeamFixture = None):
        self.currentFixture = currentInfoFixture
        self.lastInfoFixture = lastInfoFixture


class Dataset:
    def __init__(self, arrAllInfosTeamFixture: List[List[InfoTeamFixture]], arrIdsTeam: List[int],
                 idTypeReturn: int = 1):
        self.arrAllInfosTeamFixture = arrAllInfosTeamFixture
        self.arrIdsTeam = arrIdsTeam
        self.idTypeReturn = idTypeReturn

    def obterDatasetAgrupadoEmArrayAndLastFixture(self, isDadosUmLadoSo: bool = True, isOrderReverse: bool = False):
        arrAllInfosFixture = deepcopy(self.arrAllInfosTeamFixture)
        arrInfosFixtures = []
        arrInfosFixturesAgrupadas = []
        arrIdsFixturesEncontrados = []

        for idxAllInfosFixturesA in range(len(arrAllInfosFixture)):
            arrFixtures = arrAllInfosFixture[idxAllInfosFixturesA]
            for idxInfoFixtureA in range(len(arrFixtures)):
                currentFixture = arrFixtures[idxInfoFixtureA]
                lastFixture = deepcopy(arrFixtures[idxInfoFixtureA - 1])
                lastFixture.lastInfoTeamFixture = None

                if idxInfoFixtureA >= 1:
                    currentFixture.lastInfoTeamFixture = deepcopy(lastFixture)

                arrInfosFixtures.append(deepcopy(currentFixture))

        for idxInfoFixturesA in range(len(arrInfosFixtures)):
            infoFixtureA = arrInfosFixtures[idxInfoFixturesA]

            if infoFixtureA.id_fixture in arrIdsFixturesEncontrados:
                continue

            for idxInfoFixturesB in range(len(arrInfosFixtures)):
                infoFixtureB = arrInfosFixtures[idxInfoFixturesB]

                if infoFixtureA.id_fixture == infoFixtureB.id_fixture and infoFixtureA.id_team != infoFixtureB.id_team:
                    arrInfoAgrupada = []
                    arrIdsFixturesEncontrados.append(infoFixtureA.id_fixture)
                    if isDadosUmLadoSo:
                        if infoFixtureA.id_team in self.arrIdsTeam:
                            arrInfoAgrupada.append(infoFixtureA)
                            arrInfoAgrupada.append(infoFixtureB)
                        else:
                            arrInfoAgrupada.append(infoFixtureB)
                            arrInfoAgrupada.append(infoFixtureA)

                    else:
                        if infoFixtureA.is_home:
                            arrInfoAgrupada.append(infoFixtureA)
                            arrInfoAgrupada.append(infoFixtureB)
                        else:
                            arrInfoAgrupada.append(infoFixtureB)
                            arrInfoAgrupada.append(infoFixtureA)

                    if len(arrInfoAgrupada) != 2:
                        print(idxInfoFixturesA, " está se ou com muitos ", len(arrInfoAgrupada))

                    arrInfosFixturesAgrupadas.append(arrInfoAgrupada)

        arrInfosFixturesAgrupadas = sorted(arrInfosFixturesAgrupadas, key=lambda f: f[0].date_fixture,
                                           reverse=isOrderReverse)
        return arrInfosFixturesAgrupadas

    def getDatasetA(self, qtdeDados: int, isDadosUmLadoSo: bool):
        arrFixturesAgrupadas = self.obterDatasetAgrupadoEmArrayAndLastFixture(isDadosUmLadoSo=isDadosUmLadoSo,
                                                                              isOrderReverse=False)

        arrDatasetsEntradas = []
        arrDatasetsRotulos = []
        arrIdxTeamObterTreino = []
        arrIdxTeamObterPrever = []

        for idxFixtureAgrupada in range(len(arrFixturesAgrupadas)):
            teamA = arrFixturesAgrupadas[idxFixtureAgrupada][0]
            teamB = arrFixturesAgrupadas[idxFixtureAgrupada][1]

            arrA = self.getEntradaA(teamA=teamA, teamB=teamB)

            if len(arrDatasetsEntradas) == 0:
                arrDatasetsEntradas = [[] for _ in range(len(arrA))]
                arrDatasetsRotulos = [[] for _ in range(len(arrA))]

            for idxA in range(len(arrA)):
                arrDatasetsEntradas[idxA].append(arrA[idxA][0])
                arrDatasetsRotulos[idxA].append(arrA[idxA][1])

            if teamA.id_team in self.arrIdsTeam:
                if teamA.is_terminou_fulltime == 1:
                    arrIdxTeamObterTreino.append(len(arrDatasetsEntradas[0]) - 1)
                else:
                    arrIdxTeamObterPrever.append(len(arrDatasetsEntradas[0]) - 1)

            arrB = self.getEntradaA(teamA=teamB, teamB=teamA)

            for idxB in range(len(arrB)):
                arrDatasetsEntradas[idxB].append(arrB[idxB][0])
                arrDatasetsRotulos[idxB].append(arrB[idxB][1])

            if teamB.id_team in self.arrIdsTeam:
                if teamB.is_terminou_fulltime == 1:
                    arrIdxTeamObterTreino.append(len(arrDatasetsEntradas[0]) - 1)
                else:
                    arrIdxTeamObterPrever.append(len(arrDatasetsEntradas[0]) - 1)

        arrDatasetsEntradasNormalizados = [self.normalizar_z_score(ent, axis=0) for ent in arrDatasetsEntradas]

        if len(arrDatasetsEntradasNormalizados) != len(arrDatasetsRotulos):
            raise Exception("Array difecrentiados")

        arrDatasetsEntradasRetornar = [[] for _ in arrDatasetsEntradasNormalizados]
        arrDatasetsRotulosRetornar = [[] for _ in arrDatasetsEntradasNormalizados]
        arrDatasetsPreverRetornar = [[] for _ in arrDatasetsEntradasNormalizados]

        for idxType in range(len(arrDatasetsEntradasNormalizados)):
            for idx in range(len(arrDatasetsEntradasNormalizados[idxType])):
                if idx in arrIdxTeamObterTreino:
                    arrDatasetsEntradasRetornar[idxType].append(arrDatasetsEntradasNormalizados[idxType][idx])
                    arrDatasetsRotulosRetornar[idxType].append(arrDatasetsRotulos[idxType][idx])
                elif idx in arrIdxTeamObterPrever:
                    arrDatasetsPreverRetornar[idxType].append(arrDatasetsEntradasNormalizados[idxType][idx])

        qtdeDadosForTeam = qtdeDados * len(self.arrIdsTeam)
        arrEntradaCortada = [entrada[-qtdeDadosForTeam:] for entrada in arrDatasetsEntradasRetornar]
        arrRotuloCortado = [rotulo[-qtdeDadosForTeam:] for rotulo in arrDatasetsRotulosRetornar]

        return arrEntradaCortada, arrRotuloCortado, arrDatasetsPreverRetornar

    @staticmethod
    def getEntradaA(teamA: InfoTeamFixture, teamB: InfoTeamFixture):
        statsAA = [
            (teamA.media_passes_accurate
             if teamA.media_passes_accurate is not None else
             teamA.lastInfoTeamFixture.media_passes_accurate
             if teamA.lastInfoTeamFixture.media_passes_accurate is not None else 0),
            (teamA.media_ball_possession
             if teamA.media_ball_possession is not None else
             teamA.lastInfoTeamFixture.media_ball_possession
             if teamA.lastInfoTeamFixture.media_ball_possession is not None else 0),
            (teamA.media_goalkeeper_saves
             if teamA.media_goalkeeper_saves is not None else
             teamA.lastInfoTeamFixture.media_goalkeeper_saves
             if teamA.lastInfoTeamFixture.media_goalkeeper_saves is not None else 0),
            (teamA.media_corner_kicks
             if teamA.media_corner_kicks is not None else
             teamA.lastInfoTeamFixture.media_corner_kicks
             if teamA.lastInfoTeamFixture.media_corner_kicks is not None else 0)
        ]
        statsAAA = [
            (teamA.media_cards_yellow
             if teamA.media_cards_yellow is not None else
             teamA.lastInfoTeamFixture.media_cards_yellow
             if teamA.lastInfoTeamFixture.media_cards_yellow is not None else 0),
            (teamA.media_cards_red
             if teamA.media_cards_red is not None else
             teamA.lastInfoTeamFixture.media_cards_red
             if teamA.lastInfoTeamFixture.media_cards_red is not None else 0),
            (teamA.media_fouls
             if teamA.media_fouls is not None else
             teamA.lastInfoTeamFixture.media_fouls
             if teamA.lastInfoTeamFixture.media_fouls is not None else 0),
        ]
        statsAB = [
            (teamB.media_passes_accurate
             if teamB.media_passes_accurate is not None else
             teamB.lastInfoTeamFixture.media_passes_accurate
             if teamB.lastInfoTeamFixture.media_passes_accurate is not None else 0),
            (teamB.media_ball_possession
             if teamB.media_ball_possession is not None else
             teamB.lastInfoTeamFixture.media_ball_possession
             if teamB.lastInfoTeamFixture.media_ball_possession is not None else 0),
            (teamB.media_goalkeeper_saves
             if teamB.media_goalkeeper_saves is not None else
             teamB.lastInfoTeamFixture.media_goalkeeper_saves
             if teamB.lastInfoTeamFixture.media_goalkeeper_saves is not None else 0),
            (teamB.media_corner_kicks
             if teamB.media_corner_kicks is not None else
             teamB.lastInfoTeamFixture.media_corner_kicks
             if teamB.lastInfoTeamFixture.media_corner_kicks is not None else 0)
        ]
        statsAAB = [
            (teamB.media_cards_yellow
             if teamB.media_cards_yellow is not None else
             teamB.lastInfoTeamFixture.media_cards_yellow
             if teamB.lastInfoTeamFixture.media_cards_yellow is not None else 0),
            (teamB.media_cards_red
             if teamB.media_cards_red is not None else
             teamB.lastInfoTeamFixture.media_cards_red
             if teamB.lastInfoTeamFixture.media_cards_red is not None else 0),
            (teamB.media_fouls
             if teamB.media_fouls is not None else
             teamB.lastInfoTeamFixture.media_fouls
             if teamB.lastInfoTeamFixture.media_fouls is not None else 0),
        ]

        array = [
            teamA.id_team,
            teamA.date_fixture.timestamp() * 1000,
            teamA.is_home, teamA.pontos_season, teamA.saldo_goals_season,
            teamA.pontos_team, teamA.saldo_goals_team
        ]
        isMaisPontos = int(teamA.pontos_season >= teamB.pontos_season)
        isMediaVitoriaMelhor = int(teamA.media_vitorias >= teamB.media_vitorias)
        isMediaEmpateMelhor = int(teamA.media_empates >= teamB.media_empates)
        isMediaDerrotaMelhor = int(teamA.media_derrotas >= teamB.media_derrotas)
        isMarcaMaisGols = int(teamA.media_goals_fulltime >= teamB.media_goals_fulltime)
        isSofreMaisGols = int(teamA.media_goals_fulltime_conceded >= teamB.media_goals_fulltime_conceded)
        isMelhoresEstatisticasA = int(sum(statsAA) >= sum(statsAB))
        isMelhoresEstatisticasAA = int(sum(statsAAA) <= sum(statsAAB))

        array.append(isMaisPontos)
        array.append(isMediaVitoriaMelhor)
        array.append(isMediaEmpateMelhor)
        array.append(isMediaDerrotaMelhor)
        array.append(isMarcaMaisGols)
        array.append(isSofreMaisGols)
        array.append(isMelhoresEstatisticasA)
        array.append(isMelhoresEstatisticasAA)

        arrayB = [
            teamA.id_team, teamB.id_team,
            teamA.date_fixture.timestamp() * 1000,
            teamA.is_home, teamA.pontos_season, teamA.saldo_goals_season,
            teamA.pontos_team, teamA.saldo_goals_team,
            teamB.pontos_season, teamB.saldo_goals_season,
            teamB.pontos_team, teamB.saldo_goals_team
        ]
        isMaisPontosB = int(teamB.pontos_season >= teamA.pontos_season)
        isMediaVitoriaMelhorB = int(teamB.media_vitorias >= teamA.media_vitorias)
        isMediaEmpateMelhorB = int(teamB.media_empates >= teamA.media_empates)
        isMediaDerrotaMelhorB = int(teamB.media_derrotas >= teamA.media_derrotas)
        isMarcaMaisGolsB = int(teamB.media_goals_fulltime >= teamA.media_goals_fulltime)
        isSofreMaisGolsB = int(teamB.media_goals_fulltime_conceded >= teamA.media_goals_fulltime_conceded)
        isMelhoresEstatisticasAB = int(sum(statsAB) >= sum(statsAA))
        isMelhoresEstatisticasAAB = int(sum(statsAAB) <= sum(statsAAA))

        arrayB.append(isMaisPontos)
        arrayB.append(isMediaVitoriaMelhor)
        arrayB.append(isMediaEmpateMelhor)
        arrayB.append(isMediaDerrotaMelhor)
        arrayB.append(isMarcaMaisGols)
        arrayB.append(isSofreMaisGols)
        arrayB.append(isMelhoresEstatisticasA)
        arrayB.append(isMelhoresEstatisticasAA)

        arrayB.append(isMaisPontosB)
        arrayB.append(isMediaVitoriaMelhorB)
        arrayB.append(isMediaEmpateMelhorB)
        arrayB.append(isMediaDerrotaMelhorB)
        arrayB.append(isMarcaMaisGolsB)
        arrayB.append(isSofreMaisGolsB)
        arrayB.append(isMelhoresEstatisticasAB)
        arrayB.append(isMelhoresEstatisticasAAB)

        '''AA = [arrayB,
              [teamA.is_vitoria if teamA.is_home else teamB.is_vitoria]]

        BA = [arrayB,
              [int(abs(teamA.goals_fulltime - teamA.goals_fulltime_conceded) <= 1)]]'''

        AA = [array,
              [teamA.is_vitoria]]
        BA = [array,
              [int(abs(teamA.goals_fulltime - teamA.goals_fulltime_conceded) <= 1)]]

        arrs = [AA, BA]
        return arrs

    @staticmethod
    def getEntradaAB(teamA: InfoTeamFixture, teamB: InfoTeamFixture):
        statsA = [
            (teamA.media_passes_accurate
             if teamA.media_passes_accurate is not None else
             teamA.lastInfoTeamFixture.media_passes_accurate
             if teamA.lastInfoTeamFixture.media_passes_accurate is not None else 0),
            (teamA.media_ball_possession
             if teamA.media_ball_possession is not None else
             teamA.lastInfoTeamFixture.media_ball_possession
             if teamA.lastInfoTeamFixture.media_ball_possession is not None else 0),
            (teamA.media_goalkeeper_saves
             if teamA.media_goalkeeper_saves is not None else
             teamA.lastInfoTeamFixture.media_goalkeeper_saves
             if teamA.lastInfoTeamFixture.media_goalkeeper_saves is not None else 0),
            (teamA.media_corner_kicks
             if teamA.media_corner_kicks is not None else
             teamA.lastInfoTeamFixture.media_corner_kicks
             if teamA.lastInfoTeamFixture.media_corner_kicks is not None else 0)
        ]
        statsAA = [
            (teamA.media_cards_yellow
             if teamA.media_cards_yellow is not None else
             teamA.lastInfoTeamFixture.media_cards_yellow
             if teamA.lastInfoTeamFixture.media_cards_yellow is not None else 0),
            (teamA.media_cards_red
             if teamA.media_cards_red is not None else
             teamA.lastInfoTeamFixture.media_cards_red
             if teamA.lastInfoTeamFixture.media_cards_red is not None else 0),
            (teamA.media_fouls
             if teamA.media_fouls is not None else
             teamA.lastInfoTeamFixture.media_fouls
             if teamA.lastInfoTeamFixture.media_fouls is not None else 0),
        ]
        statsB = [
            (teamB.media_passes_accurate
             if teamB.media_passes_accurate is not None else
             teamB.lastInfoTeamFixture.media_passes_accurate
             if teamB.lastInfoTeamFixture.media_passes_accurate is not None else 0),
            (teamB.media_ball_possession
             if teamB.media_ball_possession is not None else
             teamB.lastInfoTeamFixture.media_ball_possession
             if teamB.lastInfoTeamFixture.media_ball_possession is not None else 0),
            (teamB.media_goalkeeper_saves
             if teamB.media_goalkeeper_saves is not None else
             teamB.lastInfoTeamFixture.media_goalkeeper_saves
             if teamB.lastInfoTeamFixture.media_goalkeeper_saves is not None else 0),
            (teamB.media_corner_kicks
             if teamB.media_corner_kicks is not None else
             teamB.lastInfoTeamFixture.media_corner_kicks
             if teamB.lastInfoTeamFixture.media_corner_kicks is not None else 0)
        ]
        statsBB = [
            (teamB.media_cards_yellow
             if teamB.media_cards_yellow is not None else
             teamB.lastInfoTeamFixture.media_cards_yellow
             if teamB.lastInfoTeamFixture.media_cards_yellow is not None else 0),
            (teamB.media_cards_red
             if teamB.media_cards_red is not None else
             teamB.lastInfoTeamFixture.media_cards_red
             if teamB.lastInfoTeamFixture.media_cards_red is not None else 0),
            (teamB.media_fouls
             if teamB.media_fouls is not None else
             teamB.lastInfoTeamFixture.media_fouls
             if teamB.lastInfoTeamFixture.media_fouls is not None else 0),
        ]
        isMelhoresEstatisticasA = int(sum(statsA) >= sum(statsB))
        isMelhoresEstatisticasAA = int(sum(statsAA) <= sum(statsBB))

        A = [[teamA.is_home,
              isMelhoresEstatisticasA, isMelhoresEstatisticasAA,
              teamA.date_fixture.timestamp() * 1000,
              int(teamA.pontos_team >= teamB.pontos_team),
              int(teamA.pontos_season >= teamB.pontos_season),
              int(teamA.saldo_goals_season >= teamB.saldo_goals_season),
              int(teamA.saldo_goals_team >= teamB.saldo_goals_team),
              int(teamA.media_vitorias >= teamB.media_vitorias),
              int(teamA.media_empates >= teamB.media_empates),
              int(teamA.media_derrotas >= teamB.media_derrotas),
              int(teamA.media_goals_halftime >= teamB.media_goals_halftime),
              int(teamA.media_goals_halftime_conceded >= teamB.media_goals_halftime_conceded),
              int(teamA.media_goals_fulltime >= teamB.media_goals_fulltime),
              int(teamA.media_goals_fulltime_conceded >= teamB.media_goals_fulltime_conceded)],
             [teamA.is_vitoria]]

        B = [[teamA.is_home,
              isMelhoresEstatisticasA, isMelhoresEstatisticasAA,
              teamA.date_fixture.timestamp() * 1000,
              int(teamA.pontos_team >= teamB.pontos_team),
              int(teamA.pontos_season >= teamB.pontos_season),
              int(teamA.saldo_goals_season >= teamB.saldo_goals_season),
              int(teamA.saldo_goals_team >= teamB.saldo_goals_team),
              int(teamA.media_vitorias >= teamB.media_vitorias),
              int(teamA.media_empates >= teamB.media_empates),
              int(teamA.media_derrotas >= teamB.media_derrotas),
              int(teamA.media_goals_halftime >= teamB.media_goals_halftime),
              int(teamA.media_goals_halftime_conceded >= teamB.media_goals_halftime_conceded),
              int(teamA.media_goals_fulltime >= teamB.media_goals_fulltime),
              int(teamA.media_goals_fulltime_conceded >= teamB.media_goals_fulltime_conceded)],
             [int(teamA.goals_fulltime - teamB.goals_fulltime_conceded >= 2)]]

        C = [[teamA.is_home,
              isMelhoresEstatisticasA, isMelhoresEstatisticasAA,
              teamA.date_fixture.timestamp() * 1000,
              int(teamA.pontos_team >= teamB.pontos_team),
              int(teamA.pontos_season >= teamB.pontos_season),
              int(teamA.saldo_goals_season >= teamB.saldo_goals_season),
              int(teamA.saldo_goals_team >= teamB.saldo_goals_team),
              int(teamA.media_vitorias >= teamB.media_vitorias),
              int(teamA.media_empates >= teamB.media_empates),
              int(teamA.media_derrotas >= teamB.media_derrotas),
              int(teamA.media_goals_halftime >= teamB.media_goals_halftime),
              int(teamA.media_goals_halftime_conceded >= teamB.media_goals_halftime_conceded),
              int(teamA.media_goals_fulltime >= teamB.media_goals_fulltime),
              int(teamA.media_goals_fulltime_conceded >= teamB.media_goals_fulltime_conceded)],
             [teamA.is_empate]]

        D = [[teamA.is_home,
              isMelhoresEstatisticasA, isMelhoresEstatisticasAA,
              teamA.date_fixture.timestamp() * 1000,
              int(teamA.pontos_team >= teamB.pontos_team),
              int(teamA.pontos_season >= teamB.pontos_season),
              int(teamA.saldo_goals_season >= teamB.saldo_goals_season),
              int(teamA.saldo_goals_team >= teamB.saldo_goals_team),
              int(teamA.media_vitorias >= teamB.media_vitorias),
              int(teamA.media_empates >= teamB.media_empates),
              int(teamA.media_derrotas >= teamB.media_derrotas),
              int(teamA.media_goals_halftime >= teamB.media_goals_halftime),
              int(teamA.media_goals_halftime_conceded >= teamB.media_goals_halftime_conceded),
              int(teamA.media_goals_fulltime >= teamB.media_goals_fulltime),
              int(teamA.media_goals_fulltime_conceded >= teamB.media_goals_fulltime_conceded)],
             [int(abs(teamA.goals_fulltime - teamB.goals_fulltime_conceded) <= 1)]]

        arrs = [A, B, C, D]
        return arrs

    @staticmethod
    def getEntradaABackup(teamA: InfoTeamFixture, teamB: InfoTeamFixture):
        A = [[teamA.id_team, teamB.id_team], [teamA.is_vitoria, teamA.is_empate, teamA.is_derrota]]
        B = [[teamA.is_home], [teamA.is_vitoria, teamA.is_empate, teamA.is_derrota]]
        C = [[teamA.pontos_team, teamB.pontos_team], [int(teamA.pontos_team >= teamB.pontos_team)]]
        D = [[teamA.pontos_season, teamB.pontos_season], [int(teamA.pontos_season >= teamB.pontos_season)]]
        E = [[teamA.saldo_goals_season, teamB.saldo_goals_season],
             [int(teamA.saldo_goals_season >= teamB.saldo_goals_season)]]
        F = [[teamA.saldo_goals_team, teamB.saldo_goals_team], [int(teamA.saldo_goals_team >= teamB.saldo_goals_team)]]
        G = [[teamA.media_vitorias, teamB.media_vitorias], [int(teamA.media_vitorias >= teamB.media_vitorias)]]
        H = [[teamA.media_empates, teamB.media_empates], [int(teamA.media_empates >= teamB.media_empates)]]
        I = [[teamA.media_derrotas, teamB.media_derrotas], [int(teamA.media_derrotas >= teamB.media_derrotas)]]
        J = [[teamA.media_goals_halftime, teamB.media_goals_halftime],
             [int(teamA.media_goals_halftime >= teamB.media_goals_halftime)]]
        K = [[teamA.media_goals_halftime_conceded, teamB.media_goals_halftime_conceded],
             [int(teamA.media_goals_halftime_conceded >= teamB.media_goals_halftime_conceded)]]
        L = [[teamA.media_goals_fulltime, teamB.media_goals_fulltime],
             [int(teamA.media_goals_fulltime >= teamB.media_goals_fulltime)]]
        M = [[teamA.media_goals_fulltime_conceded, teamB.media_goals_fulltime_conceded],
             [int(teamA.media_goals_fulltime_conceded >= teamB.media_goals_fulltime_conceded)]]

        arrs = [A, B, C, D, E, F, G, H, I, J, K, L, M]
        return arrs

    @staticmethod
    def getArrayDatasetEntradaA(currentInfoFixtureA: InfoTeamFixture, currentInfoFixtureB: InfoTeamFixture,
                                lastInfoFixtureA: InfoTeamFixture = None):
        array = [
            currentInfoFixtureA.id_team, currentInfoFixtureA.date_fixture.timestamp() * 1000,
            currentInfoFixtureA.is_home, currentInfoFixtureA.is_terminou_fulltime, currentInfoFixtureA.id_fixture,
            currentInfoFixtureA.pontos_season, currentInfoFixtureA.saldo_goals_season,
            currentInfoFixtureA.pontos_team, currentInfoFixtureA.saldo_goals_team
        ]
        isMaisPontos = int(currentInfoFixtureA.pontos_season >= currentInfoFixtureB.pontos_season)
        isMediaVitoriaMelhor = int(currentInfoFixtureA.media_vitorias >= currentInfoFixtureB.media_vitorias)
        isMediaEmpateMelhor = int(currentInfoFixtureA.media_empates >= currentInfoFixtureB.media_empates)
        isMediaDerrotaMelhor = int(currentInfoFixtureA.media_derrotas >= currentInfoFixtureB.media_derrotas)
        isMarcaMaisGols = int(currentInfoFixtureA.media_goals_fulltime >= currentInfoFixtureB.media_goals_fulltime)
        isSofreMaisGols = int(currentInfoFixtureA.media_goals_fulltime_conceded >=
                              currentInfoFixtureB.media_goals_fulltime_conceded)
        statsA = [
            (currentInfoFixtureA.media_passes_accurate
             if currentInfoFixtureA.media_passes_accurate is not None else
             lastInfoFixtureA.media_passes_accurate
             if lastInfoFixtureA.media_passes_accurate is not None else 0),
            (currentInfoFixtureA.media_ball_possession
             if currentInfoFixtureA.media_ball_possession is not None else
             lastInfoFixtureA.media_ball_possession
             if lastInfoFixtureA.media_ball_possession is not None else 0),
            (currentInfoFixtureA.media_goalkeeper_saves
             if currentInfoFixtureA.media_goalkeeper_saves is not None else
             lastInfoFixtureA.media_goalkeeper_saves
             if lastInfoFixtureA.media_goalkeeper_saves is not None else 0),
            (currentInfoFixtureA.media_corner_kicks
             if currentInfoFixtureA.media_corner_kicks is not None else
             lastInfoFixtureA.media_corner_kicks
             if lastInfoFixtureA.media_corner_kicks is not None else 0),
        ]
        statsAA = [
            (currentInfoFixtureA.media_cards_yellow
             if currentInfoFixtureA.media_cards_yellow is not None else
             lastInfoFixtureA.media_cards_yellow
             if lastInfoFixtureA.media_cards_yellow is not None else 0),
            (currentInfoFixtureA.media_cards_red
             if currentInfoFixtureA.media_cards_red is not None else
             lastInfoFixtureA.media_cards_red
             if lastInfoFixtureA.media_cards_red is not None else 0),
            (currentInfoFixtureA.media_fouls
             if currentInfoFixtureA.media_fouls is not None else
             lastInfoFixtureA.media_fouls
             if lastInfoFixtureA.media_fouls is not None else 0),
        ]
        statsB = [
            (currentInfoFixtureB.media_passes_accurate
             if currentInfoFixtureB.media_passes_accurate is not None else
             currentInfoFixtureB.lastInfoTeamFixture.media_passes_accurate
             if currentInfoFixtureB.lastInfoTeamFixture.media_passes_accurate is not None else 0),
            (currentInfoFixtureB.media_ball_possession
             if currentInfoFixtureB.media_ball_possession is not None else
             currentInfoFixtureB.lastInfoTeamFixture.media_ball_possession
             if currentInfoFixtureB.lastInfoTeamFixture.media_ball_possession is not None else 0),
            (currentInfoFixtureB.media_goalkeeper_saves
             if currentInfoFixtureB.media_goalkeeper_saves is not None else
             currentInfoFixtureB.lastInfoTeamFixture.media_goalkeeper_saves
             if currentInfoFixtureB.lastInfoTeamFixture.media_goalkeeper_saves is not None else 0),
            (currentInfoFixtureB.media_corner_kicks
             if currentInfoFixtureB.media_corner_kicks is not None else
             currentInfoFixtureB.lastInfoTeamFixture.media_corner_kicks
             if currentInfoFixtureB.lastInfoTeamFixture.media_corner_kicks is not None else 0)
        ]
        statsBB = [
            (currentInfoFixtureB.media_cards_yellow
             if currentInfoFixtureB.media_cards_yellow is not None else
             currentInfoFixtureB.lastInfoTeamFixture.media_cards_yellow
             if currentInfoFixtureB.lastInfoTeamFixture.media_cards_yellow is not None else 0),
            (currentInfoFixtureB.media_cards_red
             if currentInfoFixtureB.media_cards_red is not None else
             currentInfoFixtureB.lastInfoTeamFixture.media_cards_red
             if currentInfoFixtureB.lastInfoTeamFixture.media_cards_red is not None else 0),
            (currentInfoFixtureB.media_fouls
             if currentInfoFixtureB.media_fouls is not None else
             currentInfoFixtureB.lastInfoTeamFixture.media_fouls
             if currentInfoFixtureB.lastInfoTeamFixture.media_fouls is not None else 0),
        ]
        isMelhoresEstatisticas = sum(statsA) >= sum(statsB)
        isMelhoresEstatisticasA = sum(statsAA) <= sum(statsBB)
        array.append(isMaisPontos)
        array.append(isMediaVitoriaMelhor)
        array.append(isMediaEmpateMelhor)
        array.append(isMediaDerrotaMelhor)
        array.append(isMarcaMaisGols)
        array.append(isSofreMaisGols)
        array.append(isMelhoresEstatisticas)
        array.append(isMelhoresEstatisticasA)

        return array

    def getDatasetB(self, qtdeDados: int = None, isDadosUmLadoSo: bool = True):
        arrInfosFixturesAgrupadas = self.obterDatasetAgrupadoEmArrayAndLastFixture(isDadosUmLadoSo=isDadosUmLadoSo,
                                                                                   isOrderReverse=False)
        arrDadosEntrada = []
        arrDadosRotulo = []
        arrIndexObterArrDados = []
        for arrFixtureAgrupada in arrInfosFixturesAgrupadas:
            fixtureA = arrFixtureAgrupada[0]
            fixtureB = arrFixtureAgrupada[1]

            entradaA = self.getArrayDatasetEntradaA(currentInfoFixtureA=fixtureA, currentInfoFixtureB=fixtureB,
                                                    lastInfoFixtureA=fixtureA.lastInfoTeamFixture)
            entradaB = self.getArrayDatasetEntradaA(currentInfoFixtureA=fixtureB, currentInfoFixtureB=fixtureA,
                                                    lastInfoFixtureA=fixtureB.lastInfoTeamFixture)

            rotuloA = self.getArrayDatasetRotuloA(currentInfoFixture=fixtureA)
            rotuloB = self.getArrayDatasetRotuloA(currentInfoFixture=fixtureB)

            arrDadosEntrada.append(entradaA)
            arrDadosRotulo.append(rotuloA)
            if fixtureA.id_team in self.arrIdsTeam:
                arrIndexObterArrDados.append(len(arrDadosEntrada) - 1)

            arrDadosEntrada.append(entradaB)
            arrDadosRotulo.append(rotuloB)
            if fixtureB.id_team in self.arrIdsTeam:
                arrIndexObterArrDados.append(len(arrDadosEntrada) - 1)

        arrDadosEntradaNormalizado = self.normalizar_z_score(array=arrDadosEntrada, axis=0)
        if len(arrDadosEntradaNormalizado) != len(arrDadosRotulo):
            raise Exception("Array difecrentiados")

        arrDadosEntradaRetornar = []
        arrDadosRotuloRetornar = []
        arrDadosPreverRetornar = []

        for idxArrDados in range(len(arrDadosEntradaNormalizado)):
            if idxArrDados in arrIndexObterArrDados:
                if arrDadosEntradaNormalizado[idxArrDados][3] < 0:
                    arrDadosPreverRetornar.append(arrDadosEntradaNormalizado[idxArrDados])
                else:
                    arrDadosEntradaRetornar.append(arrDadosEntradaNormalizado[idxArrDados])
                    arrDadosRotuloRetornar.append(arrDadosRotulo[idxArrDados])

        qtdeDadosRetornar = qtdeDados * len(self.arrIdsTeam)
        return (arrDadosEntradaRetornar[-qtdeDadosRetornar:], arrDadosRotuloRetornar[-qtdeDadosRetornar:],
                arrDadosPreverRetornar)

    # Dataset agrupado com time A e B
    def getDatasetD(self, isPreverComIdsExpecificos: bool = True, qtdeDados: int = None, isDadosUmLadoSo: bool = True,
                    idTypeRotulo: int = 1):
        arrInfosFixturesAgrupadas = self.obterDatasetAgrupadoEmArrayAndLastFixture(isDadosUmLadoSo=isDadosUmLadoSo,
                                                                                   isOrderReverse=False)
        if isPreverComIdsExpecificos:
            arrEntrada, arrRotulo, arrPrever = self.getDatasetDTeamsExpecificos(
                arrInfosAgrupadas=arrInfosFixturesAgrupadas, isDadosUmLadoSo=isDadosUmLadoSo, idTypeRotulo=idTypeRotulo)

            qtdeDatas = int(len(self.arrIdsTeam) * qtdeDados)
            if qtdeDados is None:
                return arrEntrada, arrRotulo, arrPrever
            else:
                return arrEntrada[-qtdeDatas:], arrRotulo[-qtdeDatas:], arrPrever

        return [], [], []

    def getDatasetDTeamsExpecificos(self, arrInfosAgrupadas: List[List[InfoTeamFixture]], isDadosUmLadoSo,
                                    isAgruparTeams: bool = True, idTypeRotulo: int = 1):
        arrIdsExpcf = self.arrIdsTeam
        arrInfosNormalizarEntrada = []
        arrInfosNormalizarRotulo = []

        arrIdsInfosGetArrayEntrada = []
        arrIdsInfosGetArrayRotulo = []
        arrIdsTeamRotulo = []

        for idxInfoAgrupada in range(len(arrInfosAgrupadas)):
            infoHome = arrInfosAgrupadas[idxInfoAgrupada][0]
            infoAway = arrInfosAgrupadas[idxInfoAgrupada][1]

            infoEntradaHome = self.getArrayDatasetEntradaA(currentInfoFixtureA=infoHome,
                                                           currentInfoFixtureB=infoAway,
                                                           lastInfoFixtureA=infoHome.lastInfoTeamFixture)
            infoEntradaAway = self.getArrayDatasetEntradaA(currentInfoFixtureA=infoAway,
                                                           currentInfoFixtureB=infoHome,
                                                           lastInfoFixtureA=infoAway.lastInfoTeamFixture)

            if self.idTypeReturn == 1:
                infoRotuloHome = self.getArrayDatasetRotuloA(currentInfoFixture=infoHome)
                infoRotuloAway = self.getArrayDatasetRotuloA(currentInfoFixture=infoAway)
            elif self.idTypeReturn == 2:
                infoRotuloHome = self.getArrayDatasetRotuloB(currentInfoFixture=infoHome)
                infoRotuloAway = self.getArrayDatasetRotuloB(currentInfoFixture=infoAway)
            elif self.idTypeReturn == 3:
                infoRotuloHome = self.getArrayDatasetRotuloC(currentInfoFixture=infoHome)
                infoRotuloAway = self.getArrayDatasetRotuloC(currentInfoFixture=infoAway)
            elif self.idTypeReturn == 4:
                infoRotuloHome = self.getArrayDatasetRotuloF(currentInfoFixture=infoHome)
                infoRotuloAway = self.getArrayDatasetRotuloF(currentInfoFixture=infoAway)
            else:
                infoRotuloHome = self.getArrayDatasetRotuloA(currentInfoFixture=infoHome)
                infoRotuloAway = self.getArrayDatasetRotuloA(currentInfoFixture=infoAway)

            arrInfosNormalizarEntrada.append(infoEntradaHome)
            arrInfosNormalizarRotulo.append(infoRotuloHome)
            arrInfosNormalizarEntrada.append(infoEntradaAway)
            arrInfosNormalizarRotulo.append(infoRotuloAway)

            if infoHome.id_team in arrIdsExpcf and infoAway.id_team in arrIdsExpcf:
                arrIdsTeamRotulo.append(len(arrInfosNormalizarRotulo) - 2)
            elif infoHome.id_team in arrIdsExpcf:
                arrIdsTeamRotulo.append(len(arrInfosNormalizarRotulo) - 2)
            elif infoAway.id_team in arrIdsExpcf:
                arrIdsTeamRotulo.append(len(arrInfosNormalizarRotulo) - 1)

            if infoHome.id_team in arrIdsExpcf or infoAway.id_team in arrIdsExpcf:
                # -2 para pegar o Home e -1 para pegar o Away
                arrIdsInfosGetArrayEntrada.append(len(arrInfosNormalizarEntrada) - 2)
                arrIdsInfosGetArrayEntrada.append(len(arrInfosNormalizarEntrada) - 1)
                arrIdsInfosGetArrayRotulo.append(len(arrInfosNormalizarRotulo) - 2)
                arrIdsInfosGetArrayRotulo.append(len(arrInfosNormalizarRotulo) - 1)

        arrNormalizadoZScore: List[List] = self.normalizar_z_score(array=arrInfosNormalizarEntrada, axis=0)

        if isAgruparTeams:
            arrInfosAgrupadasEntradas = []
            arrInfosAgrupadasPrever = []
            arrInfosAgrupadasRotulos = []
            arrIdxRemoverDuplicados = [1, 3, 4]

            arrConcatenado = []
            for idxInfoNormalizada in range(len(arrNormalizadoZScore)):
                if idxInfoNormalizada not in arrIdsInfosGetArrayEntrada:
                    continue

                infoNormalizada = deepcopy(arrNormalizadoZScore[idxInfoNormalizada])

                if idxInfoNormalizada % 2 == 0:
                    arrConcatenado = numpy.concatenate((arrConcatenado, infoNormalizada)).tolist()
                else:
                    for idxRemover in reversed(arrIdxRemoverDuplicados):
                        infoNormalizada.pop(idxRemover)
                    arrConcatenado = numpy.concatenate((arrConcatenado, infoNormalizada)).tolist()

                    if arrConcatenado[3] < 0:
                        arrInfosAgrupadasPrever.append(arrConcatenado)
                        arrConcatenado = []
                    else:
                        arrInfosAgrupadasEntradas.append(arrConcatenado)
                        arrConcatenado = []
                        if isDadosUmLadoSo:
                            if idTypeRotulo == 1:
                                arrInfosAgrupadasRotulos.append(arrInfosNormalizarRotulo[idxInfoNormalizada - 1])
                            elif idTypeRotulo == 2:
                                arrInfosAgrupadasRotulos.append(arrInfosNormalizarRotulo[idxInfoNormalizada])
                        else:
                            if idTypeRotulo == 1:
                                arrInfosAgrupadasRotulos.append(arrInfosNormalizarRotulo[idxInfoNormalizada - 1])
                            elif idTypeRotulo == 2:
                                # Adicionar sempre o rotulo do time d fora para a previsar ficar 1 X 2
                                arrInfosAgrupadasRotulos.append(arrInfosNormalizarRotulo[idxInfoNormalizada])
                            elif idTypeRotulo == 3:
                                if (idxInfoNormalizada - 1) in arrIdsTeamRotulo:
                                    arrInfosAgrupadasRotulos.append(arrInfosNormalizarRotulo[idxInfoNormalizada - 1])
                                else:
                                    arrInfosAgrupadasRotulos.append(arrInfosNormalizarRotulo[idxInfoNormalizada])
            return arrInfosAgrupadasEntradas, arrInfosAgrupadasRotulos, arrInfosAgrupadasPrever
        return [], [], []

    @staticmethod
    def getArrayDatasetRotuloA(currentInfoFixture: InfoTeamFixture):
        isWinner = int(currentInfoFixture.goals_fulltime - currentInfoFixture.goals_fulltime_conceded >= 1)
        array = [isWinner]

        return array

    @staticmethod
    def getArrayDatasetRotuloB(currentInfoFixture: InfoTeamFixture):
        isEmpate = int(abs(currentInfoFixture.goals_fulltime - currentInfoFixture.goals_fulltime_conceded) <= 1)
        array = [isEmpate]

        return array

    @staticmethod
    def getArrayDatasetRotuloC(currentInfoFixture: InfoTeamFixture):
        isLoss = int((currentInfoFixture.goals_fulltime - currentInfoFixture.goals_fulltime_conceded) <= -1)
        array = [isLoss]

        return array

    @staticmethod
    def getArrayDatasetRotuloF(currentInfoFixture: InfoTeamFixture):
        isWinnerA = int(currentInfoFixture.goals_fulltime - currentInfoFixture.goals_fulltime_conceded >= 2)
        isWinnerB = int(currentInfoFixture.goals_fulltime - currentInfoFixture.goals_fulltime_conceded <= -2)
        isEmpate = int(abs(currentInfoFixture.goals_fulltime - currentInfoFixture.goals_fulltime_conceded) <= 1)

        array = [isEmpate]

        return array

    @staticmethod
    def getArrayDatasetEntradaE(currentInfoFixture: InfoTeamFixture, lastInfoFixture: InfoTeamFixture = None):
        array = [
            currentInfoFixture.id_team, currentInfoFixture.date_fixture.timestamp() * 1000,
            currentInfoFixture.is_home, currentInfoFixture.is_terminou_fulltime, currentInfoFixture.id_fixture,
            currentInfoFixture.pontos_season, currentInfoFixture. saldo_goals_season,
            currentInfoFixture.media_vitorias, currentInfoFixture.media_empates, currentInfoFixture.media_derrotas,
            0 if lastInfoFixture is None else lastInfoFixture.is_vitoria,
            0 if lastInfoFixture is None else lastInfoFixture.is_empate,
            0 if lastInfoFixture is None else lastInfoFixture.is_derrota,
            currentInfoFixture.media_goals_fulltime, currentInfoFixture.media_goals_fulltime_conceded,
            0 if lastInfoFixture is None else lastInfoFixture.goals_fulltime,
            0 if lastInfoFixture is None else lastInfoFixture.goals_fulltime_conceded,
            (currentInfoFixture.media_passes_accurate if currentInfoFixture.media_passes_accurate is not None else
             lastInfoFixture.media_passes_accurate if lastInfoFixture.media_passes_accurate is not None else 0),
            (currentInfoFixture.media_ball_possession if currentInfoFixture.media_ball_possession is not None else
             lastInfoFixture.media_ball_possession if lastInfoFixture.media_ball_possession is not None else 0),
            (currentInfoFixture.media_goalkeeper_saves if currentInfoFixture.media_goalkeeper_saves is not None else
             lastInfoFixture.media_goalkeeper_saves if lastInfoFixture.media_goalkeeper_saves is not None else 0),
            (currentInfoFixture.media_corner_kicks if currentInfoFixture.media_corner_kicks is not None else
             lastInfoFixture.media_corner_kicks if lastInfoFixture.media_corner_kicks is not None else 0),
        ]

        return array

    @staticmethod
    def normalizar_z_score(array, axis=0) -> list:
        arrayB = zscore(array, axis=axis)
        arrayB[numpy.isnan(arrayB)] = 0
        return arrayB.tolist()

class DatasetRegras:
    def __init__(self):
        self.ignoreIntelisence = None
        self.datasetEntrada: list = []
        self.datasetRotulo: list = []
        self.datasetPrever: list = []

    def obterDataset(self, arrIdsTeam: List[int], qtdeDadosForTeam: int = 15, limitHistorico: int = 5,
                     isObterAdversarios: bool = True,  arrIdsExpecficos: List[int] = [], idTypeReturn: int = 1,
                     isDadosUmLadoSo: bool = True, idTypeRotulo: int = 1, idTypeEntrada: int = 1):
        self.ignoreIntelisence = False
        arrIdsPrever = list(arrIdsTeam)
        arrAllInfoTeamFixture: List[List[InfoTeamFixture]] = []

        if arrIdsTeam is None or len(arrIdsTeam) == 0:
            raise Exception("e preciso passar teams")

        if isObterAdversarios:
            arrIdsTeam = self.obterIdsAdversarios(arrIdsTeam=arrIdsTeam)

        for idTeam in arrIdsTeam:
            arrInfoTeamFxture = self.obterInfoByTeam(idTeam=idTeam, limitHistorico=limitHistorico,
                                                     isGetNextGame=(idTeam in arrIdsPrever))
            ultimosJogosTeam = arrInfoTeamFxture  # arrInfoTeamFxture[-qtdeDadosForTeam:]

            if len(ultimosJogosTeam) < qtdeDadosForTeam + 1:
                if ultimosJogosTeam[0].id_team in arrIdsPrever:
                    raise Exception("Sem dados suficientes para esses time. " + ultimosJogosTeam[0].name_team +
                                    str(len(ultimosJogosTeam)))

            arrAllInfoTeamFixture.append(ultimosJogosTeam)

        if idTypeEntrada == 1:
            datasetEntrada, datasetRotulo, datasetPrever = (
                Dataset(arrAllInfosTeamFixture=arrAllInfoTeamFixture, arrIdsTeam=arrIdsExpecficos,
                        idTypeReturn=idTypeReturn).getDatasetD(
                    qtdeDados=qtdeDadosForTeam, isDadosUmLadoSo=isDadosUmLadoSo, idTypeRotulo=idTypeRotulo))
        elif idTypeEntrada == 2:
            datasetEntrada, datasetRotulo, datasetPrever = (
                Dataset(arrAllInfosTeamFixture=arrAllInfoTeamFixture, arrIdsTeam=arrIdsExpecficos,
                        idTypeReturn=idTypeReturn).getDatasetB(
                    qtdeDados=qtdeDadosForTeam, isDadosUmLadoSo=isDadosUmLadoSo))
        elif idTypeEntrada == 4:
            datasetEntrada, datasetRotulo, datasetPrever = (
                Dataset(arrAllInfosTeamFixture=arrAllInfoTeamFixture, arrIdsTeam=arrIdsExpecficos,
                        idTypeReturn=idTypeReturn).getDatasetA(
                    qtdeDados=qtdeDadosForTeam, isDadosUmLadoSo=isDadosUmLadoSo))
        else:
            datasetEntrada, datasetRotulo, datasetPrever = [], [], []

        return datasetEntrada, datasetRotulo, datasetPrever

    def obterInfoByTeam(self, idTeam: int, limitHistorico: int = 5, isGetNextGame: bool = True) -> List[InfoTeamFixture]:
        self.ignoreIntelisence = True
        arrStatusFinished = ["FT", "AET", "PEN"]
        fixtureModel = FixturesModel(teamsModel=TeamsModel())
        team: Team = fixtureModel.teamsModel.obterByColumnsID(arrDados=[idTeam])[0]
        arrFixtures: List[Fixture] = fixtureModel.obterFixturesOrderDataBy(id_team=idTeam)
        if isGetNextGame:
            nextFixture: Fixture = fixtureModel.obterNextFixtureByidSeasonTeam(id_team=idTeam)[0]
            arrFixtures.append(nextFixture)
        arrInfoTeamFixture: List[InfoTeamFixture] = []

        for fixture in arrFixtures:
            infoFixture = InfoTeamFixture()
            infoFixture.date_fixture = fixture.date
            infoFixture.id_season = fixture.id_season
            infoFixture.id_fixture = fixture.id
            infoFixture.id_team = idTeam
            infoFixture.is_statistics_fixture = fixture.has_statistics_fixture
            infoFixture.status_fixture = fixture.status
            infoFixture.time_elapsed = fixture.time_elapsed
            infoFixture.is_terminou_fulltime = int(fixture.status in arrStatusFinished)
            infoFixture.is_terminou_after_fulltime = int((infoFixture.is_terminou_fulltime and
                                                          fixture.status != "FT"))
            infoFixture.name_team = team.name

            fixtureTeam: FixtureTeams = fixtureModel.fixturesTeamsModel.obterByColumns(
                arrNameColuns=["id_fixture", "id_team"], arrDados=[fixture.id, team.id])[0]

            if fixtureTeam.id_team != idTeam:
                raise Exception("UEPPAAAA ids dos times diferente " + str(fixtureTeam.id_team) + " / " + str(idTeam))

            infoFixture.winner = self.normalizarWinner(is_winner=fixtureTeam.is_winner)
            infoFixture.is_home = fixtureTeam.is_home
            infoFixture.is_vitoria = int(infoFixture.winner == 2)
            infoFixture.is_empate = int(infoFixture.winner == 1)
            infoFixture.is_derrota = int(infoFixture.winner == 0)

            infoFixture.goals_total = fixtureTeam.goals
            infoFixture.goals_halftime = fixtureTeam.goals_halftime
            infoFixture.goals_fulltime = fixtureTeam.goals_fulltime
            infoFixture.goals_extratime = fixtureTeam.goals_extratime
            infoFixture.goals_penalty = fixtureTeam.goals_penalty

            infoFixture.gols_primeiro_tempo = fixtureTeam.goals_halftime
            infoFixture.gols_primeiro_tempo_conceded = fixtureTeam.goals_halftime_conceded

            infoFixture.gols_segundo_tempo = fixtureTeam.goals_fulltime - fixtureTeam.goals_halftime
            infoFixture.gols_segundo_tempo_conceded = (
                    fixtureTeam.goals_fulltime_conceded - fixtureTeam.goals_halftime_conceded)

            infoFixture.goals_total_conceded = fixtureTeam.goals_conceded
            infoFixture.goals_halftime_conceded = fixtureTeam.goals_halftime_conceded
            infoFixture.goals_fulltime_conceded = fixtureTeam.goals_fulltime_conceded
            infoFixture.goals_extratime_conceded = fixtureTeam.goals_extratime_conceded
            infoFixture.goals_penalty_conceded = fixtureTeam.goals_penalty_conceded

            if infoFixture.is_statistics_fixture:
                arrStatistics: List[FixtureTeamStatistic] = fixtureModel.fixturesTeamsStatisticsModel.obterByColumns(
                    arrNameColuns=["id_fixture", "id_team"], arrDados=[fixture.id, team.id])

                if len(arrStatistics) == 0:
                    raise Exception("Diz que tem statistics mas nao tem, verifique id fixture" + str(fixture.id))

                infoFixture = self.adicionarStatisticsInfoTeam(arrStatistics=arrStatistics, infoTeam=infoFixture)

            arrInfoTeamFixture.append(infoFixture)

        self.calcularInfosTeamFixture(arrInfoTeamFixture=arrInfoTeamFixture, limitHistorico=limitHistorico)
        return arrInfoTeamFixture

    def obterIdsAdversarios(self, arrIdsTeam: List[int]):
        self.ignoreIntelisence = True
        arrAllIds = list(arrIdsTeam)

        for idTeam in arrIdsTeam:
            fixtureModel = FixturesModel(teamsModel=TeamsModel())
            arrFixtures: List[Fixture] = fixtureModel.obterFixturesOrderDataBy(id_team=idTeam)

            for fixture in arrFixtures:
                arrFixturesTeam: List[FixtureTeams] = fixtureModel.fixturesTeamsModel.obterByColumns(
                    arrNameColuns=["id_fixture"], arrDados=[fixture.id])

                for fixtureTeam in arrFixturesTeam:
                    if fixtureTeam.id_team not in arrAllIds:
                        arrAllIds.append(fixtureTeam.id_team)

        return arrAllIds

    def normalizarWinner(self, is_winner) -> int:
        self.ignoreIntelisence = True
        if is_winner is None:
            isWinner = 1
        elif is_winner == 0:
            isWinner = 0
        elif is_winner == 1:
            isWinner = 2
        else:
            raise Exception("Não foi possivel normalizar os dados de winner")

        return isWinner

    def adicionarStatisticsInfoTeam(self, arrStatistics: List[FixtureTeamStatistic], infoTeam: InfoTeamFixture):
        self.ignoreIntelisence = True

        for statistic in arrStatistics:
            if statistic.id_type_statistic == EnumTypeStatistics.shots_on_goal.value:
                infoTeam.shots_on_goal = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.shots_off_goal.value:
                infoTeam.shots_off_goal = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.shots_total.value:
                infoTeam.shots_total = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.shots_bloqued.value:
                infoTeam.shots_bloqued = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.fouls.value:
                infoTeam.fouls = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.corner_kicks.value:
                infoTeam.corner_kicks = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.offsides.value:
                infoTeam.offsides = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.ball_possession.value:
                infoTeam.ball_possession = statistic.value * 100  # Pois é um float 0.algo
            elif statistic.id_type_statistic == EnumTypeStatistics.cards_yellow.value:
                infoTeam.cards_yellow = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.cards_red.value:
                infoTeam.cards_red = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.goalkeeper_saves.value:
                infoTeam.goalkeeper_saves = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.passes_total.value:
                infoTeam.passes_total = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.passes_accurate.value:
                infoTeam.passes_accurate = statistic.value
            elif statistic.id_type_statistic == EnumTypeStatistics.passes_precision.value:
                infoTeam.passes_precision = statistic.value * 100  # Pois é um float 0.algo
            else:
                # 5 - Shots insidebox
                # 6 - Shots outsidebox
                # 17 - expected_goals
                arrIdsIgnorar = [5, 6, 17]

                if statistic.id_type_statistic in arrIdsIgnorar:
                    continue

                raise Exception("Parece q temos uma nova statistica nao cadastrada " +
                                str(infoTeam.id_fixture) + " " + str(statistic.id))
        return infoTeam

    def calcularInfosTeamFixture(self, arrInfoTeamFixture: List[InfoTeamFixture], limitHistorico: int = 5):
        self.calcularDadosSeason(arrInfoTeamFixture=arrInfoTeamFixture)
        self.calcularDadosMediaTeam(arrInfoTeamFixture=arrInfoTeamFixture, limitHistorico=limitHistorico)

    def calcularDadosSeason(self, arrInfoTeamFixture: List[InfoTeamFixture]):
        self.ignoreIntelisence = True
        arrDictSeason = []

        for idxInfoFixture in range(len(arrInfoTeamFixture)):
            infoFixture = arrInfoTeamFixture[idxInfoFixture]
            isEntrouDict = False

            dictSeason = {
                "id_season": infoFixture.id_season,
                "pontos": 0,
                "saldo_gols": 0
            }

            for idxDictSeason in range(len(arrDictSeason)):
                if arrDictSeason[idxDictSeason]["id_season"] == infoFixture.id_season:
                    dictSeason = arrDictSeason[idxDictSeason]
                    isEntrouDict = True

            infoFixture.pontos_season = dictSeason["pontos"]
            infoFixture.saldo_goals_season = dictSeason["saldo_gols"]

            if infoFixture.is_vitoria:
                dictSeason["pontos"] += 3
            elif infoFixture.is_empate:
                dictSeason["pontos"] += 1

            dictSeason["saldo_gols"] += infoFixture.goals_fulltime
            dictSeason["saldo_gols"] -= infoFixture.goals_fulltime_conceded

            if not isEntrouDict:
                arrDictSeason.append(dictSeason)

        return arrInfoTeamFixture

    def calcularDadosMediaTeam(self, arrInfoTeamFixture: List[InfoTeamFixture], limitHistorico: int = 5):
        self.ignoreIntelisence = True
        arrInfosHistory: List[InfoTeamFixture] = []
        arrInfosHistoryHome: List[InfoTeamFixture] = []
        arrInfosHistoryAway: List[InfoTeamFixture] = []
        arrInfosHistoryStatistics: List[InfoTeamFixture] = []

        backupDictInfoMedias = {
            "pontos_team": 0.,
            "saldo_goals_team": 0.,
            "media_gols_primeiro_tempo": 0.,
            "media_gols_primeiro_tempo_conceded": 0.,
            "media_gols_segundo_tempo": 0.,
            "media_gols_segundo_tempo_conceded": 0.,
            "media_goals_total": 0.,
            "media_goals_total_conceded": 0.,
            "media_goals_halftime": 0.,
            "media_goals_halftime_conceded": 0.,
            "media_goals_fulltime": 0.,
            "media_goals_fulltime_conceded": 0.,

            "media_vitorias": 0.,
            "media_empates": 0.,
            "media_derrotas": 0.
        }
        backupDictInfoMediasHome = {
            "media_gols_primeiro_tempo_home": 0.,
            "media_gols_primeiro_tempo_conceded_home": 0.,
            "media_gols_segundo_tempo_home": 0.,
            "media_gols_segundo_tempo_conceded_home": 0.,
            "media_goals_total_home": 0.,
            "media_goals_total_conceded_home": 0.,
            "media_goals_halftime_home": 0.,
            "media_goals_halftime_conceded_home": 0.,
            "media_goals_fulltime_home": 0.,
            "media_goals_fulltime_conceded_home": 0.,

            "media_vitorias_home": 0.,
            "media_empates_home": 0.,
            "media_derrotas_home": 0.,
        }
        backupDictInfoMediasAway = {
            "media_gols_primeiro_tempo_away": 0.,
            "media_gols_primeiro_tempo_conceded_away": 0.,
            "media_gols_segundo_tempo_away": 0.,
            "media_gols_segundo_tempo_conceded_away": 0.,
            "media_goals_total_away": 0.,
            "media_goals_total_conceded_away": 0.,
            "media_goals_halftime_away": 0.,
            "media_goals_halftime_conceded_away": 0.,
            "media_goals_fulltime_away": 0.,
            "media_goals_fulltime_conceded_away": 0.,

            "media_vitorias_away": 0.,
            "media_empates_away": 0.,
            "media_derrotas_away": 0.,
        }
        backupDictInfoMediasStatistics = {
            "media_shots_on_goal": 0.,
            "media_shots_off_goal": 0.,
            "media_shots_total": 0.,
            "media_shots_bloqued": 0.,
            "media_fouls": 0.,
            "media_corner_kicks": 0.,
            "media_offsides": 0.,
            "media_ball_possession": 0.,
            "media_cards_yellow": 0.,
            "media_cards_red": 0.,
            "media_goalkeeper_saves": 0.,
            "media_passes_total": 0.,
            "media_passes_accurate": 0.,
            "media_passes_precision": 0.
        }

        dictInfoMedias = {
            "pontos_team": 0.,
            "saldo_goals_team": 0.,
            "media_gols_primeiro_tempo": 0.,
            "media_gols_primeiro_tempo_conceded": 0.,
            "media_gols_segundo_tempo": 0.,
            "media_gols_segundo_tempo_conceded": 0.,
            "media_goals_total": 0.,
            "media_goals_total_conceded": 0.,
            "media_goals_halftime": 0.,
            "media_goals_halftime_conceded": 0.,
            "media_goals_fulltime": 0.,
            "media_goals_fulltime_conceded": 0.,

            "media_vitorias": 0.,
            "media_empates": 0.,
            "media_derrotas": 0.
        }
        dictInfoMediasHome = {
            "media_gols_primeiro_tempo_home": 0.,
            "media_gols_primeiro_tempo_conceded_home": 0.,
            "media_gols_segundo_tempo_home": 0.,
            "media_gols_segundo_tempo_conceded_home": 0.,
            "media_goals_total_home": 0.,
            "media_goals_total_conceded_home": 0.,
            "media_goals_halftime_home": 0.,
            "media_goals_halftime_conceded_home": 0.,
            "media_goals_fulltime_home": 0.,
            "media_goals_fulltime_conceded_home": 0.,

            "media_vitorias_home": 0.,
            "media_empates_home": 0.,
            "media_derrotas_home": 0.,
        }
        dictInfoMediasAway = {
            "media_gols_primeiro_tempo_away": 0.,
            "media_gols_primeiro_tempo_conceded_away": 0.,
            "media_gols_segundo_tempo_away": 0.,
            "media_gols_segundo_tempo_conceded_away": 0.,
            "media_goals_total_away": 0.,
            "media_goals_total_conceded_away": 0.,
            "media_goals_halftime_away": 0.,
            "media_goals_halftime_conceded_away": 0.,
            "media_goals_fulltime_away": 0,
            "media_goals_fulltime_conceded_away": 0.,

            "media_vitorias_away": 0.,
            "media_empates_away": 0.,
            "media_derrotas_away": 0.,
        }
        dictInfoMediasStatistics = {
            "media_shots_on_goal": 0.,
            "media_shots_off_goal": 0.,
            "media_shots_total": 0.,
            "media_shots_bloqued": 0.,
            "media_fouls": 0.,
            "media_corner_kicks": 0.,
            "media_offsides": 0.,
            "media_ball_possession": 0.,
            "media_cards_yellow": 0.,
            "media_cards_red": 0.,
            "media_goalkeeper_saves": 0.,
            "media_passes_total": 0.,
            "media_passes_accurate": 0.,
            "media_passes_precision": 0.
        }

        for idxInfoFixture in range(len(arrInfoTeamFixture)):
            infoTeamFixture = arrInfoTeamFixture[idxInfoFixture]

            infoTeamFixture.pontos_team = dictInfoMedias["pontos_team"]
            infoTeamFixture.saldo_goals_team = dictInfoMedias["saldo_goals_team"]

            infoTeamFixture.media_gols_primeiro_tempo = dictInfoMedias["media_gols_primeiro_tempo"]
            infoTeamFixture.media_gols_primeiro_tempo_conceded = dictInfoMedias["media_gols_primeiro_tempo_conceded"]
            infoTeamFixture.media_gols_segundo_tempo = dictInfoMedias["media_gols_segundo_tempo"]
            infoTeamFixture.media_gols_segundo_tempo_conceded = dictInfoMedias["media_gols_segundo_tempo_conceded"]
            infoTeamFixture.media_goals_total = dictInfoMedias["media_goals_total"]
            infoTeamFixture.media_goals_total_conceded = dictInfoMedias["media_goals_total_conceded"]
            infoTeamFixture.media_goals_halftime = dictInfoMedias["media_goals_halftime"]
            infoTeamFixture.media_goals_halftime_conceded = dictInfoMedias["media_goals_halftime_conceded"]
            infoTeamFixture.media_goals_fulltime = dictInfoMedias["media_goals_fulltime"]
            infoTeamFixture.media_goals_fulltime_conceded = dictInfoMedias["media_goals_fulltime_conceded"]
            infoTeamFixture.media_vitorias = dictInfoMedias["media_vitorias"]
            infoTeamFixture.media_empates = dictInfoMedias["media_empates"]
            infoTeamFixture.media_derrotas = dictInfoMedias["media_derrotas"]

            # Infos das Statistics
            infoTeamFixture.media_shots_on_goal = dictInfoMediasStatistics["media_shots_on_goal"]
            infoTeamFixture.media_shots_off_goal = dictInfoMediasStatistics["media_shots_off_goal"]
            infoTeamFixture.media_shots_total = dictInfoMediasStatistics["media_shots_total"]
            infoTeamFixture.media_shots_bloqued = dictInfoMediasStatistics["media_shots_bloqued"]
            infoTeamFixture.media_fouls = dictInfoMediasStatistics["media_fouls"]
            infoTeamFixture.media_corner_kicks = dictInfoMediasStatistics["media_corner_kicks"]
            infoTeamFixture.media_offsides = dictInfoMediasStatistics["media_offsides"]
            infoTeamFixture.media_ball_possession = dictInfoMediasStatistics["media_ball_possession"]
            infoTeamFixture.media_cards_yellow = dictInfoMediasStatistics["media_cards_yellow"]
            infoTeamFixture.media_cards_red = dictInfoMediasStatistics["media_cards_red"]
            infoTeamFixture.media_goalkeeper_saves = dictInfoMediasStatistics["media_goalkeeper_saves"]
            infoTeamFixture.media_passes_total = dictInfoMediasStatistics["media_passes_total"]
            infoTeamFixture.media_passes_accurate = dictInfoMediasStatistics["media_passes_accurate"]
            infoTeamFixture.media_passes_precision = dictInfoMediasStatistics["media_passes_precision"]

            # Infos quando joga HOME
            infoTeamFixture.media_gols_primeiro_tempo_home = dictInfoMediasHome["media_gols_primeiro_tempo_home"]
            infoTeamFixture.media_gols_primeiro_tempo_conceded_home = dictInfoMediasHome["media_gols_primeiro_tempo_conceded_home"]
            infoTeamFixture.media_gols_segundo_tempo_home = dictInfoMediasHome["media_gols_segundo_tempo_home"]
            infoTeamFixture.media_gols_segundo_tempo_conceded_home = dictInfoMediasHome["media_gols_segundo_tempo_conceded_home"]
            infoTeamFixture.media_goals_total_home = dictInfoMediasHome["media_goals_total_home"]
            infoTeamFixture.media_goals_total_conceded_home = dictInfoMediasHome["media_goals_total_conceded_home"]
            infoTeamFixture.media_goals_halftime_home = dictInfoMediasHome["media_goals_halftime_home"]
            infoTeamFixture.media_goals_halftime_conceded_home = dictInfoMediasHome["media_goals_halftime_conceded_home"]
            infoTeamFixture.media_goals_fulltime_home = dictInfoMediasHome["media_goals_fulltime_home"]
            infoTeamFixture.media_goals_fulltime_conceded_home = dictInfoMediasHome["media_goals_fulltime_conceded_home"]
            infoTeamFixture.media_vitorias_home = dictInfoMediasHome["media_vitorias_home"]
            infoTeamFixture.media_empates_home = dictInfoMediasHome["media_empates_home"]
            infoTeamFixture.media_derrotas_home = dictInfoMediasHome["media_derrotas_home"]

            # Infos quando joga AWAY
            infoTeamFixture.media_gols_primeiro_tempo_away = dictInfoMediasAway["media_gols_primeiro_tempo_away"]
            infoTeamFixture.media_gols_primeiro_tempo_conceded_away = dictInfoMediasAway["media_gols_primeiro_tempo_conceded_away"]
            infoTeamFixture.media_gols_segundo_tempo_away = dictInfoMediasAway["media_gols_segundo_tempo_away"]
            infoTeamFixture.media_gols_segundo_tempo_conceded_away = dictInfoMediasAway["media_gols_segundo_tempo_conceded_away"]
            infoTeamFixture.media_goals_total_away = dictInfoMediasAway["media_goals_total_away"]
            infoTeamFixture.media_goals_total_conceded_away = dictInfoMediasAway["media_goals_total_conceded_away"]
            infoTeamFixture.media_goals_halftime_away = dictInfoMediasAway["media_goals_halftime_away"]
            infoTeamFixture.media_goals_halftime_conceded_away = dictInfoMediasAway["media_goals_halftime_conceded_away"]
            infoTeamFixture.media_goals_fulltime_away = dictInfoMediasAway["media_goals_fulltime_away"]
            infoTeamFixture.media_goals_fulltime_conceded_away = dictInfoMediasAway["media_goals_fulltime_conceded_away"]
            infoTeamFixture.media_vitorias_away = dictInfoMediasAway["media_vitorias_away"]
            infoTeamFixture.media_empates_away = dictInfoMediasAway["media_empates_away"]
            infoTeamFixture.media_derrotas_away = dictInfoMediasAway["media_derrotas_away"]

            if len(arrInfosHistory) == limitHistorico:
                arrInfosHistory.pop(0)
                arrInfosHistory.append(infoTeamFixture)
            elif len(arrInfosHistory) > limitHistorico:
                raise Exception("OPSSS mais dados q o limite")
            else:
                arrInfosHistory.append(infoTeamFixture)

            dictInfoMedias = deepcopy(backupDictInfoMedias)
            for infoHistory in arrInfosHistory:
                if infoHistory.is_vitoria == 1:
                    dictInfoMedias["pontos_team"] += 3
                elif infoHistory.is_empate == 1:
                    dictInfoMedias["pontos_team"] += 1

                dictInfoMedias["saldo_goals_team"] += infoHistory.goals_fulltime
                dictInfoMedias["saldo_goals_team"] -= infoHistory.goals_fulltime_conceded

                dictInfoMedias["media_gols_primeiro_tempo"] += infoHistory.gols_primeiro_tempo
                dictInfoMedias["media_gols_primeiro_tempo_conceded"] += infoHistory.gols_primeiro_tempo_conceded
                dictInfoMedias["media_gols_segundo_tempo"] += infoHistory.gols_segundo_tempo
                dictInfoMedias["media_gols_segundo_tempo_conceded"] += infoHistory.gols_segundo_tempo_conceded
                dictInfoMedias["media_goals_total"] += infoHistory.goals_total
                dictInfoMedias["media_goals_total_conceded"] += infoHistory.goals_total_conceded
                dictInfoMedias["media_goals_halftime"] += infoHistory.goals_halftime
                dictInfoMedias["media_goals_halftime_conceded"] += infoHistory.goals_halftime_conceded
                dictInfoMedias["media_goals_fulltime"] += infoHistory.goals_fulltime
                dictInfoMedias["media_goals_fulltime_conceded"] += infoHistory.goals_fulltime_conceded
                dictInfoMedias["media_vitorias"] += infoHistory.is_vitoria
                dictInfoMedias["media_empates"] += infoHistory.is_empate
                dictInfoMedias["media_derrotas"] += infoHistory.is_derrota

            dictInfoMedias["media_gols_primeiro_tempo"] = (
                    dictInfoMedias["media_gols_primeiro_tempo"] / len(arrInfosHistory))
            dictInfoMedias["media_gols_primeiro_tempo_conceded"] = (
                    dictInfoMedias["media_gols_primeiro_tempo_conceded"] / len(arrInfosHistory))
            dictInfoMedias["media_gols_segundo_tempo"] = (
                    dictInfoMedias["media_gols_segundo_tempo"] / len(arrInfosHistory))
            dictInfoMedias["media_gols_segundo_tempo_conceded"] = (
                    dictInfoMedias["media_gols_segundo_tempo_conceded"] / len(arrInfosHistory))
            dictInfoMedias["media_goals_total"] = (
                    dictInfoMedias["media_goals_total"] / len(arrInfosHistory))
            dictInfoMedias["media_goals_total_conceded"] = (
                    dictInfoMedias["media_goals_total_conceded"] / len(arrInfosHistory))
            dictInfoMedias["media_goals_halftime"] = (
                    dictInfoMedias["media_goals_halftime"] / len(arrInfosHistory))
            dictInfoMedias["media_goals_halftime_conceded"] = (
                    dictInfoMedias["media_goals_halftime_conceded"] / len(arrInfosHistory))
            dictInfoMedias["media_goals_fulltime"] = (
                    dictInfoMedias["media_goals_fulltime"] / len(arrInfosHistory))
            dictInfoMedias["media_goals_fulltime_conceded"] = (
                    dictInfoMedias["media_goals_fulltime_conceded"] / len(arrInfosHistory))
            dictInfoMedias["media_vitorias"] = (
                    dictInfoMedias["media_vitorias"] / len(arrInfosHistory))
            dictInfoMedias["media_empates"] = (
                    dictInfoMedias["media_empates"] / len(arrInfosHistory))
            dictInfoMedias["media_derrotas"] = (
                    dictInfoMedias["media_derrotas"] / len(arrInfosHistory))

            # Infos quando joga Home
            if infoTeamFixture.is_home:
                if len(arrInfosHistoryHome) == limitHistorico:
                    arrInfosHistoryHome.pop(0)
                    arrInfosHistoryHome.append(infoTeamFixture)
                elif len(arrInfosHistoryHome) > limitHistorico:
                    raise Exception("OPSSS mais dados q o limite")
                else:
                    arrInfosHistoryHome.append(infoTeamFixture)

                dictInfoMediasHome = deepcopy(backupDictInfoMediasHome)
                for infoHistory in arrInfosHistoryHome:
                    dictInfoMediasHome["media_gols_primeiro_tempo_home"] += infoHistory.gols_primeiro_tempo
                    dictInfoMediasHome["media_gols_primeiro_tempo_conceded_home"] += infoHistory.gols_primeiro_tempo_conceded
                    dictInfoMediasHome["media_gols_segundo_tempo_home"] += infoHistory.gols_segundo_tempo
                    dictInfoMediasHome["media_gols_segundo_tempo_conceded_home"] += infoHistory.gols_segundo_tempo_conceded
                    dictInfoMediasHome["media_goals_total_home"] += infoHistory.goals_total
                    dictInfoMediasHome["media_goals_total_conceded_home"] += infoHistory.goals_total_conceded
                    dictInfoMediasHome["media_goals_halftime_home"] += infoHistory.goals_halftime
                    dictInfoMediasHome["media_goals_halftime_conceded_home"] += infoHistory.goals_halftime_conceded
                    dictInfoMediasHome["media_goals_fulltime_home"] += infoHistory.goals_fulltime
                    dictInfoMediasHome["media_goals_fulltime_conceded_home"] += infoHistory.goals_fulltime_conceded
                    dictInfoMediasHome["media_vitorias_home"] += infoHistory.is_vitoria
                    dictInfoMediasHome["media_empates_home"] += infoHistory.is_empate
                    dictInfoMediasHome["media_derrotas_home"] += infoHistory.is_derrota

                dictInfoMediasHome["media_gols_primeiro_tempo_home"] = (
                        dictInfoMediasHome["media_gols_primeiro_tempo_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_gols_primeiro_tempo_conceded_home"] = (
                        dictInfoMediasHome["media_gols_primeiro_tempo_conceded_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_gols_segundo_tempo_home"] = (
                        dictInfoMediasHome["media_gols_segundo_tempo_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_gols_segundo_tempo_conceded_home"] = (
                        dictInfoMediasHome["media_gols_segundo_tempo_conceded_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_goals_total_home"] = (
                        dictInfoMediasHome["media_goals_total_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_goals_total_conceded_home"] = (
                        dictInfoMediasHome["media_goals_total_conceded_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_goals_halftime_home"] = (
                        dictInfoMediasHome["media_goals_halftime_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_goals_halftime_conceded_home"] = (
                        dictInfoMediasHome["media_goals_halftime_conceded_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_goals_fulltime_home"] = (
                        dictInfoMediasHome["media_goals_fulltime_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_goals_fulltime_conceded_home"] = (
                        dictInfoMediasHome["media_goals_fulltime_conceded_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_vitorias_home"] = (
                        dictInfoMediasHome["media_vitorias_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_empates_home"] = (
                        dictInfoMediasHome["media_empates_home"] / len(arrInfosHistoryHome))
                dictInfoMediasHome["media_derrotas_home"] = (
                        dictInfoMediasHome["media_derrotas_home"] / len(arrInfosHistoryHome))

            # Infos quando joga Away
            if not infoTeamFixture.is_home:
                if len(arrInfosHistoryAway) == limitHistorico:
                    arrInfosHistoryAway.pop(0)
                    arrInfosHistoryAway.append(infoTeamFixture)
                elif len(arrInfosHistoryAway) > limitHistorico:
                    raise Exception("OPSSS mais dados q o limite")
                else:
                    arrInfosHistoryAway.append(infoTeamFixture)

                dictInfoMediasAway = deepcopy(backupDictInfoMediasAway)
                for infoHistory in arrInfosHistoryAway:
                    dictInfoMediasAway["media_gols_primeiro_tempo_away"] += infoHistory.gols_primeiro_tempo
                    dictInfoMediasAway["media_gols_primeiro_tempo_conceded_away"] += infoHistory.gols_primeiro_tempo_conceded
                    dictInfoMediasAway["media_gols_segundo_tempo_away"] += infoHistory.gols_segundo_tempo
                    dictInfoMediasAway["media_gols_segundo_tempo_conceded_away"] += infoHistory.gols_segundo_tempo_conceded
                    dictInfoMediasAway["media_goals_total_away"] += infoHistory.goals_total
                    dictInfoMediasAway["media_goals_total_conceded_away"] += infoHistory.goals_total_conceded
                    dictInfoMediasAway["media_goals_halftime_away"] += infoHistory.goals_halftime
                    dictInfoMediasAway["media_goals_halftime_conceded_away"] += infoHistory.goals_halftime_conceded
                    dictInfoMediasAway["media_goals_fulltime_away"] += infoHistory.goals_fulltime
                    dictInfoMediasAway["media_goals_fulltime_conceded_away"] += infoHistory.goals_fulltime_conceded
                    dictInfoMediasAway["media_vitorias_away"] += infoHistory.is_vitoria
                    dictInfoMediasAway["media_empates_away"] += infoHistory.is_empate
                    dictInfoMediasAway["media_derrotas_away"] += infoHistory.is_derrota

                dictInfoMediasAway["media_gols_primeiro_tempo_away"] = (
                        dictInfoMediasAway["media_gols_primeiro_tempo_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_gols_primeiro_tempo_conceded_away"] = (
                        dictInfoMediasAway["media_gols_primeiro_tempo_conceded_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_gols_segundo_tempo_away"] = (
                        dictInfoMediasAway["media_gols_segundo_tempo_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_gols_segundo_tempo_conceded_away"] = (
                        dictInfoMediasAway["media_gols_segundo_tempo_conceded_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_goals_total_away"] = (
                        dictInfoMediasAway["media_goals_total_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_goals_total_conceded_away"] = (
                        dictInfoMediasAway["media_goals_total_conceded_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_goals_halftime_away"] = (
                        dictInfoMediasAway["media_goals_halftime_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_goals_halftime_conceded_away"] = (
                        dictInfoMediasAway["media_goals_halftime_conceded_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_goals_fulltime_away"] = (
                        dictInfoMediasAway["media_goals_fulltime_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_goals_fulltime_conceded_away"] = (
                        dictInfoMediasAway["media_goals_fulltime_conceded_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_vitorias_away"] = (
                        dictInfoMediasAway["media_vitorias_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_empates_away"] = (
                        dictInfoMediasAway["media_empates_away"] / len(arrInfosHistoryAway))
                dictInfoMediasAway["media_derrotas_away"] = (
                        dictInfoMediasAway["media_derrotas_away"] / len(arrInfosHistoryAway))

            # Infos Statistics
            if infoTeamFixture.is_statistics_fixture:
                if len(arrInfosHistoryStatistics) == limitHistorico:
                    arrInfosHistoryStatistics.pop(0)
                    arrInfosHistoryStatistics.append(infoTeamFixture)
                elif len(arrInfosHistoryStatistics) > limitHistorico:
                    raise Exception("OPSSS mais dados q o limite")
                else:
                    arrInfosHistoryStatistics.append(infoTeamFixture)

                dictInfoMediasStatistics = deepcopy(backupDictInfoMediasStatistics)
                for infoHistory in arrInfosHistoryStatistics:
                    dictInfoMediasStatistics["media_shots_on_goal"] += infoHistory.shots_on_goal
                    dictInfoMediasStatistics["media_shots_off_goal"] += infoHistory.shots_off_goal
                    dictInfoMediasStatistics["media_shots_total"] += infoHistory.shots_total
                    dictInfoMediasStatistics["media_shots_bloqued"] += infoHistory.shots_bloqued
                    dictInfoMediasStatistics["media_fouls"] += infoHistory.fouls
                    dictInfoMediasStatistics["media_corner_kicks"] += infoHistory.corner_kicks
                    dictInfoMediasStatistics["media_offsides"] += infoHistory.offsides
                    dictInfoMediasStatistics["media_ball_possession"] += infoHistory.ball_possession
                    dictInfoMediasStatistics["media_cards_yellow"] += infoHistory.cards_yellow
                    dictInfoMediasStatistics["media_cards_red"] += infoHistory.cards_red
                    dictInfoMediasStatistics["media_goalkeeper_saves"] += infoHistory.goalkeeper_saves
                    dictInfoMediasStatistics["media_passes_total"] += infoHistory.passes_total
                    dictInfoMediasStatistics["media_passes_accurate"] += infoHistory.passes_accurate
                    dictInfoMediasStatistics["media_passes_precision"] += infoHistory.passes_precision

                dictInfoMediasStatistics["media_shots_on_goal"] = (
                        dictInfoMediasStatistics["media_shots_on_goal"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_shots_off_goal"] = (
                        dictInfoMediasStatistics["media_shots_off_goal"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_shots_total"] = (
                        dictInfoMediasStatistics["media_shots_total"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_shots_bloqued"] = (
                        dictInfoMediasStatistics["media_shots_bloqued"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_fouls"] = (
                        dictInfoMediasStatistics["media_fouls"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_corner_kicks"] = (
                        dictInfoMediasStatistics["media_corner_kicks"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_offsides"] = (
                        dictInfoMediasStatistics["media_offsides"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_ball_possession"] = (
                        dictInfoMediasStatistics["media_ball_possession"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_cards_yellow"] = (
                        dictInfoMediasStatistics["media_cards_yellow"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_cards_red"] = (
                        dictInfoMediasStatistics["media_cards_red"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_goalkeeper_saves"] = (
                        dictInfoMediasStatistics["media_goalkeeper_saves"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_passes_total"] = (
                        dictInfoMediasStatistics["media_passes_total"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_passes_accurate"] = (
                        dictInfoMediasStatistics["media_passes_accurate"] / len(arrInfosHistoryStatistics))
                dictInfoMediasStatistics["media_passes_precision"] = (
                        dictInfoMediasStatistics["media_passes_precision"] / len(arrInfosHistoryStatistics))

        return arrInfoTeamFixture

