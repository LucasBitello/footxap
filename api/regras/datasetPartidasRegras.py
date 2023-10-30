import math

import numpy

from api.models.fixturesModel import Fixture
from api.models.fixturesTeamsModel import FixtureTeams

from api.regras.fixturesRegras import FixturesRegras
from api.regras.iaUteisRegras import IAUteisRegras


class DatasetPartidaRotulo:

    def __init__(self):
        # winner deve ser 0 vitoria casa, 1 empate, 2 vitoria fora
        self.winner: int = None
        self.chance_empate: int = None
        self.chance_vitoria: int = None
        self.is_superior: int = None
        self.winner_home: int = None
        self.winner_away: int = None
        self.gols_home: int = 0
        self.gols_away: int = 0
        self.is_winner_home: int = 0
        self.is_winner_away: int = 0

        self.vitoria_home: int = 0
        self.empate_home: int = 0
        self.derrota_home: int = 0

        self.vitoria_away: int = 0
        self.empate_away: int = 0
        self.derrota_away: int = 0

        self.is_superior_home: int = 0
        self.is_superior_away: int = 0

        self.chance_empate_home: int = 0
        self.chance_empate_away: int = 0

        self.chance_vitoria_home: int = 0
        self.chance_vitoria_away: int = 0

        self.superior: int = 0
        self.superior_home: int = 0
        self.superior_away: int = 0
        self.vitoria: int = 0

        self.winner_by_gols: int = 0
        self.winner_by_gols_home: int = 0
        self.winner_by_gols_away: int = 0
        self.qtde_gols_partida: int = 0
        self.qtde_gols_partida_home: int = 0
        self.qtde_gols_partida_away: int = 0
        self.is_mais_dois_meio: int = 0
        self.is_home_marca: int = 0
        self.is_away_marca: int = 0
        self.is_amba_marca: int = 0
        self.is_none_marca: int = 0
        self.isMenosDeMeio: int = 0
        self.isMaisDeMeio: int = 0
        self.isPar: int = 0
        self.isMaisDeUmEMeioHome = 0
        self.isMaisDeUmEMeioAway = 0



class DatasetPartidaEntrada:
    def __init__(self):
        # informações identificação time não usar no treinamento
        self.data_partida: str = None

        # Informaçoes da partida
        self.is_prever: bool = False
        self.id_season: int = None

        self.dataset_partida_rotulo = DatasetPartidaRotulo()

        # informações para treinamento Team Home
        self.id_team_home: int = None
        self.season_pontos_home: int = 0
        self.season_saldo_gols_home: int = 0
        self.team_pontos_home: int = 0
        self.team_saldo_gols_home: int = 0
        self.season_media_gols_marcados_home: int = 0
        self.season_media_gols_sofridos_home: int = 0
        self.team_media_gols_marcados_home: int = 0
        self.team_media_gols_sofridos_home: int = 0

        self.team_empates_casa_home: int = 0
        self.team_vitorias_casa_home: int = 0
        self.team_derrotas_casa_home: int = 0

        self.team_empates_fora_home: int = 0
        self.team_vitorias_fora_home: int = 0
        self.team_derrotas_fora_home: int = 0

        self.team_ultimos_empates_home: int = 0
        self.team_ultimos_vitorias_home: int = 0
        self.team_ultimos_derrotas_home: int = 0

        self.team_max_empates_home: int = 0
        self.team_max_vitorias_home: int = 0
        self.team_max_derrotas_home: int = 0
        # media deve ficar em um range de 0 a 2 sendo 0 derrota 1 empate e 2 vitoria
        self.team_media_resultados_partida_home: float = 0.0
        self.team_melhorando_resultados_partida_home: int = None

        # informações para treinamento Team Away
        self.id_team_away: int = None
        self.season_pontos_away: int = 0
        self.season_saldo_gols_away: int = 0
        self.team_pontos_away: int = 0
        self.team_saldo_gols_away: int = 0
        self.season_media_gols_marcados_away: int = 0
        self.season_media_gols_sofridos_away: int = 0
        self.team_media_gols_marcados_away: int = 0
        self.team_media_gols_sofridos_away: int = 0

        self.team_empates_casa_away: int = 0
        self.team_vitorias_casa_away: int = 0
        self.team_derrotas_casa_away: int = 0

        self.team_empates_fora_away: int = 0
        self.team_vitorias_fora_away: int = 0
        self.team_derrotas_fora_away: int = 0

        self.team_ultimos_empates_away: int = 0
        self.team_ultimos_vitorias_away: int = 0
        self.team_ultimos_derrotas_away: int = 0

        self.team_max_empates_away: int = 0
        self.team_max_vitorias_away: int = 0
        self.team_max_derrotas_away: int = 0

        # media deve ficar em um range de 0 a 2 sendo 0 derrota 1 empate e 2 vitoria
        self.team_media_resultados_partida_away: float = 0.0
        self.team_melhorando_resultados_partida_away: int = None

        # Infos jogos passados
        self.team_jogo_gols_marcados_home: int = 0
        self.team_jogo_gols_sofridos_home: int = 0
        self.team_jogo_isPar_home: int = 0
        self.team_jogo_winner_by_gols_home: int = 0
        self.team_jogo_isSuperior_home: int = 0
        self.team_jogo_isMenosDeMeio_home: int = 0
        self.team_jogo_winnerHome: int = 0

        self.team_jogo_gols_marcados_away: int = 0
        self.team_jogo_gols_sofridos_away: int = 0
        self.team_jogo_isPar_away: int = 0
        self.team_jogo_winner_by_gols_away: int = 0
        self.team_jogo_isSuperior_away: int = 0
        self.team_jogo_isMenosDeMeio_away: int = 0
        self.team_jogo_winnerAway: int = 0

    def obterDadoEntradaEmArray(self, arridsTeam: list = None, id_jogo_home: int = 0, isAgrupasTeams: bool = True,
                                isPassadaTempoDupla: bool = True):
        isHomePontuacaoMaiorAway = 1 if self.season_pontos_home >= self.season_pontos_away else 0
        isAwayPontuacaoMaiorHome = 1 if self.season_pontos_away > self.season_pontos_home else 0
        isJogandoEmCasa = 1 if self.id_team_home in arridsTeam else 0
        isHome = 1 if self.id_team_home in arridsTeam else 0

        arrDados = []
        if isAgrupasTeams:
            arrDados.append(
                [self.id_team_home, 1, id_jogo_home, isHomePontuacaoMaiorAway, self.team_jogo_winnerHome,
                 self.season_pontos_home, self.season_saldo_gols_home,
                 self.team_pontos_home, self.team_saldo_gols_home,
                 self.season_media_gols_marcados_home, self.season_media_gols_sofridos_home,
                 self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                 self.team_jogo_gols_marcados_home, self.team_jogo_gols_sofridos_home,
                 self.team_jogo_isSuperior_home, self.team_jogo_isMenosDeMeio_home,
                 self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home,
                 self.team_derrotas_casa_home, self.team_empates_casa_home, self.team_vitorias_casa_home,
                 self.team_ultimos_derrotas_home, self.team_ultimos_empates_home, self.team_ultimos_vitorias_home]
            )

            arrDados.append(
                [self.id_team_away, 0, id_jogo_home, isAwayPontuacaoMaiorHome, self.team_jogo_winnerAway,
                 self.season_pontos_away, self.season_saldo_gols_away,
                 self.team_pontos_away, self.team_saldo_gols_away,
                 self.season_media_gols_marcados_away, self.season_media_gols_sofridos_away,
                 self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                 self.team_jogo_gols_marcados_away, self.team_jogo_gols_sofridos_away,
                 self.team_jogo_isSuperior_away, self.team_jogo_isMenosDeMeio_away,
                 self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away,
                 self.team_derrotas_fora_away, self.team_empates_fora_away, self.team_vitorias_fora_away,
                 self.team_ultimos_derrotas_away, self.team_ultimos_empates_away, self.team_ultimos_vitorias_away]
            )
        else:
            if isPassadaTempoDupla:
                if self.id_team_away in arridsTeam:
                    arrDados.append(
                        [self.id_team_away, 0, id_jogo_home,

                         isAwayPontuacaoMaiorHome, self.team_jogo_winnerAway,
                         self.season_pontos_away, self.season_saldo_gols_away,
                         self.team_pontos_away, self.team_saldo_gols_away,
                         self.season_media_gols_marcados_away, self.season_media_gols_sofridos_away,
                         self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                         self.team_jogo_gols_marcados_away, self.team_jogo_gols_sofridos_away,
                         self.team_jogo_isSuperior_away, self.team_jogo_isMenosDeMeio_away,
                         self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away,
                         self.team_derrotas_fora_away, self.team_empates_fora_away, self.team_vitorias_fora_away,
                         self.team_ultimos_derrotas_away, self.team_ultimos_empates_away,
                         self.team_ultimos_vitorias_away,
                         self.id_team_home]
                    )
                    arrDados.append(
                        [self.id_team_home, 1, id_jogo_home,

                         isHomePontuacaoMaiorAway, self.team_jogo_winnerHome,
                         self.season_pontos_home, self.season_saldo_gols_home,
                         self.team_pontos_home, self.team_saldo_gols_home,
                         self.season_media_gols_marcados_home, self.season_media_gols_sofridos_home,
                         self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                         self.team_jogo_gols_marcados_home, self.team_jogo_gols_sofridos_home,
                         self.team_jogo_isSuperior_home, self.team_jogo_isMenosDeMeio_home,
                         self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home,
                         self.team_derrotas_casa_home, self.team_empates_casa_home, self.team_vitorias_casa_home,
                         self.team_ultimos_derrotas_home, self.team_ultimos_empates_home,
                         self.team_ultimos_vitorias_home,
                         self.id_team_away]
                    )
                else:
                    arrDados.append(
                        [self.id_team_home, 1, id_jogo_home,

                         isHomePontuacaoMaiorAway, self.team_jogo_winnerHome,
                         self.season_pontos_home, self.season_saldo_gols_home,
                         self.team_pontos_home, self.team_saldo_gols_home,
                         self.season_media_gols_marcados_home, self.season_media_gols_sofridos_home,
                         self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                         self.team_jogo_gols_marcados_home, self.team_jogo_gols_sofridos_home,
                         self.team_jogo_isSuperior_home, self.team_jogo_isMenosDeMeio_home,
                         self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home,
                         self.team_derrotas_casa_home, self.team_empates_casa_home, self.team_vitorias_casa_home,
                         self.team_ultimos_derrotas_home, self.team_ultimos_empates_home, self.team_ultimos_vitorias_home,
                         self.id_team_away]
                    )

                    arrDados.append(
                        [self.id_team_away, 0, id_jogo_home,

                         isAwayPontuacaoMaiorHome, self.team_jogo_winnerAway,
                         self.season_pontos_away, self.season_saldo_gols_away,
                         self.team_pontos_away, self.team_saldo_gols_away,
                         self.season_media_gols_marcados_away, self.season_media_gols_sofridos_away,
                         self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                         self.team_jogo_gols_marcados_away, self.team_jogo_gols_sofridos_away,
                         self.team_jogo_isSuperior_away, self.team_jogo_isMenosDeMeio_away,
                         self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away,
                         self.team_derrotas_fora_away, self.team_empates_fora_away, self.team_vitorias_fora_away,
                         self.team_ultimos_derrotas_away, self.team_ultimos_empates_away, self.team_ultimos_vitorias_away,
                         self.id_team_home]
                    )
            else:
                arrDados.append(
                    [self.id_team_home, 1, id_jogo_home,

                     isHomePontuacaoMaiorAway, self.team_jogo_winnerHome,
                     self.season_pontos_home, self.season_saldo_gols_home,
                     self.team_pontos_home, self.team_saldo_gols_home,
                     self.season_media_gols_marcados_home, self.season_media_gols_sofridos_home,
                     self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                     self.team_jogo_gols_marcados_home, self.team_jogo_gols_sofridos_home,
                     self.team_jogo_isSuperior_home, self.team_jogo_isMenosDeMeio_home,
                     self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home,
                     self.team_derrotas_casa_home, self.team_empates_casa_home, self.team_vitorias_casa_home,
                     self.team_ultimos_derrotas_home, self.team_ultimos_empates_home, self.team_ultimos_vitorias_home,

                     isAwayPontuacaoMaiorHome, self.team_jogo_winnerAway,
                     self.season_pontos_away, self.season_saldo_gols_away,
                     self.team_pontos_away, self.team_saldo_gols_away,
                     self.season_media_gols_marcados_away, self.season_media_gols_sofridos_away,
                     self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                     self.team_jogo_gols_marcados_away, self.team_jogo_gols_sofridos_away,
                     self.team_jogo_isSuperior_away, self.team_jogo_isMenosDeMeio_away,
                     self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away,
                     self.team_derrotas_fora_away, self.team_empates_fora_away, self.team_vitorias_fora_away,
                     self.team_ultimos_derrotas_away, self.team_ultimos_empates_away, self.team_ultimos_vitorias_away]
                )

                arrDados.append(
                    [self.id_team_away, 0, id_jogo_home,

                     isAwayPontuacaoMaiorHome, self.team_jogo_winnerAway,
                     self.season_pontos_away, self.season_saldo_gols_away,
                     self.team_pontos_away, self.team_saldo_gols_away,
                     self.season_media_gols_marcados_away, self.season_media_gols_sofridos_away,
                     self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                     self.team_jogo_gols_marcados_away, self.team_jogo_gols_sofridos_away,
                     self.team_jogo_isSuperior_away, self.team_jogo_isMenosDeMeio_away,
                     self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away,
                     self.team_derrotas_fora_away, self.team_empates_fora_away, self.team_vitorias_fora_away,
                     self.team_ultimos_derrotas_away, self.team_ultimos_empates_away, self.team_ultimos_vitorias_away,

                     isHomePontuacaoMaiorAway, self.team_jogo_winnerHome,
                     self.season_pontos_home, self.season_saldo_gols_home,
                     self.team_pontos_home, self.team_saldo_gols_home,
                     self.season_media_gols_marcados_home, self.season_media_gols_sofridos_home,
                     self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                     self.team_jogo_gols_marcados_home, self.team_jogo_gols_sofridos_home,
                     self.team_jogo_isSuperior_home, self.team_jogo_isMenosDeMeio_home,
                     self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home,
                     self.team_derrotas_casa_home, self.team_empates_casa_home, self.team_vitorias_casa_home,
                     self.team_ultimos_derrotas_home, self.team_ultimos_empates_home, self.team_ultimos_vitorias_home]
                )

        return arrDados

    def obterDadoRotulosEmArray(self, arridsTeam: list = None, isGruparTeams: bool = True, idTypeReturn: int = 1):
        # cada array interno representa uma camada de saida.
        arrDados = []
        if isGruparTeams:
            if idTypeReturn == 1:
                arrDados = [[[self.dataset_partida_rotulo.superior_away]]]
            elif idTypeReturn == 2:
                arrDados = [[[self.dataset_partida_rotulo.winner_away]]]
            elif idTypeReturn == 3:
                arrDados = [[[self.dataset_partida_rotulo.chance_vitoria_home,
                              self.dataset_partida_rotulo.chance_empate_home,
                              self.dataset_partida_rotulo.chance_empate_away,
                              self.dataset_partida_rotulo.chance_vitoria_away]]]
            elif idTypeReturn == 4:
                arrDados = [[[self.dataset_partida_rotulo.winner_away]]]
            elif idTypeReturn == 5:
                arrDados = [[[self.dataset_partida_rotulo.isMenosDeMeio]]]
            elif idTypeReturn == 6:
                '''arrDados = [[[self.dataset_partida_rotulo.derrota_home,
                              self.dataset_partida_rotulo.chance_empate,
                              self.dataset_partida_rotulo.vitoria_home,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioHome,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioAway]]]'''

                arr = [0 for i in range(5)]
                arr[self.dataset_partida_rotulo.winner_by_gols_home] = 1
                arrDados = [[arr]]
            elif idTypeReturn == 7:
                arr = [0 for i in range(5)]
                arr[self.dataset_partida_rotulo.winner_by_gols_away] = 1
                arrDados = [[arr]]
        else:
            if idTypeReturn == 1:
                arrDados = [[[self.dataset_partida_rotulo.superior_home]],
                            [[self.dataset_partida_rotulo.superior_away]]]
            elif idTypeReturn == 2:
                arrDados = [[[self.dataset_partida_rotulo.winner_home]],
                            [[self.dataset_partida_rotulo.winner_away]]]
            elif idTypeReturn == 3:
                arrDados = [[[self.dataset_partida_rotulo.chance_vitoria_home,
                              self.dataset_partida_rotulo.chance_empate_home,
                              self.dataset_partida_rotulo.chance_empate_away,
                              self.dataset_partida_rotulo.chance_vitoria_away]],
                            [[self.dataset_partida_rotulo.chance_vitoria_home,
                              self.dataset_partida_rotulo.chance_empate_home,
                              self.dataset_partida_rotulo.chance_empate_away,
                              self.dataset_partida_rotulo.chance_vitoria_away]]]
            elif idTypeReturn == 4:
                if self.id_team_away in arridsTeam:
                    arrDados = [[[self.dataset_partida_rotulo.winner_away]],
                                [[self.dataset_partida_rotulo.winner_away]]]
                else:
                    arrDados = [[[self.dataset_partida_rotulo.winner_home]],
                                [[self.dataset_partida_rotulo.winner_home]]]
            elif idTypeReturn == 5:
                if self.id_team_away in arridsTeam:
                    arrDados = [[[self.dataset_partida_rotulo.winner_by_gols_away]],
                                [[self.dataset_partida_rotulo.winner_by_gols_away]]]
                else:
                    arrDados = [[[self.dataset_partida_rotulo.winner_by_gols_home]],
                                [[self.dataset_partida_rotulo.winner_by_gols_home]]]
            elif idTypeReturn == 6:
                arrDados = [[[self.dataset_partida_rotulo.derrota_home,
                              self.dataset_partida_rotulo.chance_empate,
                              self.dataset_partida_rotulo.vitoria_home,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioHome,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioAway]],
                            [[self.dataset_partida_rotulo.derrota_away,
                              self.dataset_partida_rotulo.chance_empate,
                              self.dataset_partida_rotulo.vitoria_away,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioAway,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioHome]]]
            elif idTypeReturn == 7:
                arrDados = [[[self.dataset_partida_rotulo.derrota_away,
                              self.dataset_partida_rotulo.chance_empate,
                              self.dataset_partida_rotulo.vitoria_away,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioAway,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioHome]],
                            [[self.dataset_partida_rotulo.derrota_home,
                              self.dataset_partida_rotulo.chance_empate,
                              self.dataset_partida_rotulo.vitoria_home,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioHome,
                              self.dataset_partida_rotulo.isMaisDeUmEMeioAway]]]
        return arrDados


class DatasetPartidasRegras:
    def obter(self, arrIdsTeam: list, limitHistoricoMedias: int = 10, isNormalizarSaidaEmClasse: bool = True,
              isFiltrarTeams: bool = True, qtdeDados: int = 40, isAgruparTeams: bool = True,
              idTypeReturn: int = 1, isPassadaTempoDupla: bool = True):
        iaUteisRegras = IAUteisRegras()

        arrDatasetEntradaEmArray = []
        arrDatasetRotuloEmArray = []
        arrIndexDadosRemover = []
        arrDatasetPartida = self.obterInfosDatabase(arrIdsTeam=arrIdsTeam, limitHistoricoMedias=limitHistoricoMedias)
        arrIndexDados = []

        indexPrev = 0
        index = -1
        for i in range(len(arrDatasetPartida)):
            '''if isFiltrarTeams:
                if (arrDatasetPartida[i].id_team_home not in arrIdsTeam and
                        arrDatasetPartida[i].id_team_away not in arrIdsTeam):
                    continue'''

            index += 1
            for retEntrada in arrDatasetPartida[i].obterDadoEntradaEmArray(arridsTeam=arrIdsTeam,
                                                                           id_jogo_home=index,
                                                                           isAgrupasTeams=isAgruparTeams,
                                                                           isPassadaTempoDupla=isPassadaTempoDupla):
                arrDatasetEntradaEmArray.append(retEntrada)

                if arrDatasetPartida[i].is_prever == 1:
                    if ((index not in arrIndexDadosRemover and isAgruparTeams) or
                            ((retEntrada[0] in arrIdsTeam or (retEntrada[-1] in arrIdsTeam and isPassadaTempoDupla))
                             and indexPrev not in arrIndexDadosRemover and not isAgruparTeams)):
                        arrIndexDadosRemover.append(index if isAgruparTeams else indexPrev)

                if isFiltrarTeams:
                    # if arrDatasetPartida[index].id_team_away in arrIdsTeam or
                    #     arrDatasetPartida[index].id_team_home in arrIdsTeam:
                    if retEntrada[0] in arrIdsTeam or (retEntrada[-1] in arrIdsTeam and isPassadaTempoDupla):
                        if arrDatasetPartida[i].is_prever != 1:
                            if isAgruparTeams:
                                arrIndexDados.append(index)
                            else:
                                arrIndexDados.append(indexPrev)

                indexPrev += 1

            for retRotulo in arrDatasetPartida[i].obterDadoRotulosEmArray(arridsTeam=arrIdsTeam,
                                                                          isGruparTeams=isAgruparTeams,
                                                                          idTypeReturn=idTypeReturn):
                arrDatasetRotuloEmArray.append(retRotulo)

        if len(arrDatasetRotuloEmArray) != len(arrDatasetEntradaEmArray) and not isAgruparTeams:
            raise Exception("opss, array diferentes")

        arrDatasetEntradaEmArrayNormalizado, max_value_entrada, min_value_entrada = \
            iaUteisRegras.normalizar_dataset(arrDatasetEntradaEmArray)

        arrDatasetRotuloEmArrayNormalizado = arrDatasetRotuloEmArray
        """arrDatasetRotuloEmArrayNormalizado, max_value_rotulo, min_value_rotulo = \
            iaUteisRegras.normalizar_dataset(arrDatasetRotuloEmArray)"""

        arrDatasetPreverEmArrayNormalizado = []
        newArrEntradasNormalizadas = []
        newEntradaNormalizada = []
        newArrRotulosNormalizados = []
        newRotulosNormalizados = []

        if isAgruparTeams:
            for indexEntradaNormalizada in range(len(arrDatasetEntradaEmArrayNormalizado)):
                if (indexEntradaNormalizada + 1) % 2 != 0:
                    newEntradaNormalizada = arrDatasetEntradaEmArrayNormalizado[indexEntradaNormalizada]
                    # newRotulosNormalizados = arrDatasetRotuloEmArrayNormalizado[indexEntradaNormalizada]
                else:
                    newEntradaNormalizada = (
                        numpy.concatenate((newEntradaNormalizada,
                                           arrDatasetEntradaEmArrayNormalizado[indexEntradaNormalizada])).tolist())

                    # newRotulosNormalizados = (
                        # numpy.concatenate((newRotulosNormalizados,
                                           # arrDatasetRotuloEmArrayNormalizado[indexEntradaNormalizada])).tolist())

                    newArrEntradasNormalizadas.append(newEntradaNormalizada)
                    # newArrRotulosNormalizados.append(newRotulosNormalizados)
                    newEntradaNormalizada = []
                    newRotulosNormalizados = []

        for indexRemover in reversed(arrIndexDadosRemover):
            if isAgruparTeams:
                elementoPrever = newArrEntradasNormalizadas.pop(indexRemover)
                # newArrRotulosNormalizados.pop(indexRemover)
                arrDatasetRotuloEmArrayNormalizado.pop(indexRemover)
            else:
                elementoPrever = arrDatasetEntradaEmArrayNormalizado.pop(indexRemover)
                arrDatasetRotuloEmArrayNormalizado.pop(indexRemover)
            arrDatasetPreverEmArrayNormalizado.append(elementoPrever)

        arrDatasetPreverEmArrayNormalizado = arrDatasetPreverEmArrayNormalizado[::-1]

        if isNormalizarSaidaEmClasse:
            if isAgruparTeams:
                # newArrRotulosNormalizados = self.normalizarSaidaEmClasses(newArrRotulosNormalizados)
                arrDatasetRotuloEmArrayNormalizado = self.normalizarSaidaEmClasses(arrDatasetRotuloEmArrayNormalizado)
            else:
                arrDatasetRotuloEmArrayNormalizado = self.normalizarSaidaEmClasses(arrDatasetRotuloEmArrayNormalizado)
        else:
            if False:
                arrsNormalizados = []
                for i in range(len(newArrRotulosNormalizados)):
                    normalizado = [[]]
                    for j in newArrRotulosNormalizados[i]:
                        normalizado[0].append(j[0])
                    arrsNormalizados.append(normalizado)
                newArrRotulosNormalizados = arrsNormalizados
            else:
                arrsNormalizados = []
                """for i in range(len(arrDatasetRotuloEmArrayNormalizado)):
                    normalizado = [[]]
                    for j in arrDatasetRotuloEmArrayNormalizado[i]:
                        normalizado[0].append(j[0])
                    arrsNormalizados.append(normalizado)
                arrDatasetRotuloEmArrayNormalizado = arrsNormalizados"""

        arrEntradas = []
        arrRotulos = []

        if isFiltrarTeams:
            for index in arrIndexDados:
                if isAgruparTeams:
                    arrEntradas.append(newArrEntradasNormalizadas[index])
                    # arrRotulos.append(newArrRotulosNormalizados[index])
                    arrRotulos.append(arrDatasetRotuloEmArrayNormalizado[index])
                else:
                    arrEntradas.append(arrDatasetEntradaEmArrayNormalizado[index])
                    arrRotulos.append(arrDatasetRotuloEmArrayNormalizado[index])
        else:
            if isAgruparTeams:
                arrEntradas = newArrEntradasNormalizadas
                # arrRotulos = newArrRotulosNormalizados
                arrRotulos = arrDatasetRotuloEmArrayNormalizado
            else:
                arrEntradas = arrDatasetEntradaEmArrayNormalizado
                arrRotulos = arrDatasetRotuloEmArrayNormalizado

        return arrEntradas[-qtdeDados:], arrRotulos[-qtdeDados:], arrDatasetPreverEmArrayNormalizado

    @staticmethod
    def normalizarSaidaEmClasses(arrRotulos: list):
        arrIndexCamadaValuesSaida = [[] for i in range(len(arrRotulos[0]))]

        for indexCamadaSaida in range(len(arrIndexCamadaValuesSaida)):
            for i in range(len(arrRotulos[0][indexCamadaSaida])):
                arrIndexCamadaValuesSaida[indexCamadaSaida].append([])

        for dadoRotulo in arrRotulos:
            for indexCamada in range(len(arrIndexCamadaValuesSaida)):
                for indexValue in range(len(arrIndexCamadaValuesSaida[indexCamada])):
                    if dadoRotulo[indexCamada][indexValue] not in arrIndexCamadaValuesSaida[indexCamada][indexValue]:
                        arrIndexCamadaValuesSaida[indexCamada][indexValue].append(dadoRotulo[indexCamada][indexValue])
                        arrIndexCamadaValuesSaida[indexCamada][indexValue] = sorted(
                            arrIndexCamadaValuesSaida[indexCamada][indexValue])
            pass

        rotulosNormalizadosEmClasse = []
        for dadoRotulo in arrRotulos:
            arrCamadaNormalizada = []
            for indexCamada in range(len(arrIndexCamadaValuesSaida)):
                arrDadosClasseNormalizado = []
                for indexValue in range(len(arrIndexCamadaValuesSaida[indexCamada])):
                    arrDadosValueNormalizado = [0 for i in arrIndexCamadaValuesSaida[indexCamada][indexValue]]

                    if len(arrDadosValueNormalizado) <= 2:
                        arrDadosValueNormalizado = [dadoRotulo[indexCamada][indexValue]]
                    else:
                        indexValueIgual = arrIndexCamadaValuesSaida[indexCamada][indexValue].index(
                            dadoRotulo[indexCamada][indexValue])
                        arrDadosValueNormalizado[indexValueIgual] = 1

                    for valueClasses in arrDadosValueNormalizado:
                        arrDadosClasseNormalizado.append(valueClasses)
                arrCamadaNormalizada.append(arrDadosClasseNormalizado)
            rotulosNormalizadosEmClasse.append(arrCamadaNormalizada)
        return rotulosNormalizadosEmClasse

    def obterInfosDatabase(self, arrIdsTeam: list, limitHistoricoMedias):
        fixtureRegras = FixturesRegras()

        arrFixtures = fixtureRegras.obterFixturesByIdsTeam(arrIdsTeam=arrIdsTeam)
        arrDatasetPartida: list[DatasetPartidaEntrada] = []
        arrKeysFinished = ['FT', 'AET', 'PEN']
        arrKeysWarning = ['CANC']

        for fixture in arrFixtures:
            newDatasetPartida = DatasetPartidaEntrada()
            newDatasetPartida.is_prever = 0 if fixture.status in arrKeysFinished else 1

            if fixture.status in arrKeysWarning:
                print("######## Warning status em perigo aaaaaaaa: \n", fixture.__dict__)
                continue

            if newDatasetPartida.is_prever == 1:
                print("vai prever: \n", fixture.__dict__)

            newDatasetPartida.data_partida = fixture.date
            newDatasetPartida.id_season = fixture.id_season

            fixture.teams: list[FixtureTeams] = fixture.teams

            for fixtureTeam in fixture.teams:
                if fixtureTeam.is_home == 1:
                    # Time jogando em casa
                    newDatasetPartida.id_team_home = fixtureTeam.id_team
                    newDatasetPartida.dataset_partida_rotulo.gols_home += fixtureTeam.goals
                    newDatasetPartida.dataset_partida_rotulo.winner = self.getWinnerNormalizadoPartida(
                        is_winner=fixtureTeam.is_winner, is_home=fixtureTeam.is_home)

                    newDatasetPartida.dataset_partida_rotulo.winner_home = self.getWinnerNormalizado(
                        is_winner=fixtureTeam.is_winner)

                    newDatasetPartida.dataset_partida_rotulo.is_winner_home = \
                        (1 if newDatasetPartida.dataset_partida_rotulo.winner_home == 2 else 0)

                    newDatasetPartida.dataset_partida_rotulo.vitoria_home = \
                        1 if newDatasetPartida.dataset_partida_rotulo.winner_home == 2 else 0
                    newDatasetPartida.dataset_partida_rotulo.empate_home = \
                        1 if newDatasetPartida.dataset_partida_rotulo.winner_home == 1 else 0
                    newDatasetPartida.dataset_partida_rotulo.derrota_home = \
                        1 if newDatasetPartida.dataset_partida_rotulo.winner_home == 0 else 0

                    newDatasetPartida.season_pontos_home, newDatasetPartida.season_saldo_gols_home = \
                        self.obterPontuacaoTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                id_season=newDatasetPartida.id_season)

                    newDatasetPartida.team_pontos_home, newDatasetPartida.team_saldo_gols_home = \
                        self.obterPontuacaoTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida)

                    newDatasetPartida.season_media_gols_marcados_home, newDatasetPartida.season_media_gols_sofridos_home = \
                        self.obterMediaGolsTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                id_season=newDatasetPartida.id_season)

                    newDatasetPartida.team_media_gols_marcados_home, newDatasetPartida.team_media_gols_sofridos_home = \
                        self.obterMediaGolsTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                limitDados=limitHistoricoMedias)

                    newDatasetPartida.team_media_resultados_partida_home, \
                        newDatasetPartida.team_melhorando_resultados_partida_home = \
                        self.obterMediaResultadosPartida(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                         limitDados=limitHistoricoMedias)

                    newDatasetPartida.team_empates_casa_home, newDatasetPartida.team_vitorias_casa_home, \
                        newDatasetPartida.team_derrotas_casa_home, newDatasetPartida.team_empates_fora_home, \
                        newDatasetPartida.team_vitorias_fora_home, newDatasetPartida.team_derrotas_fora_home = \
                        self.obterInfoCasaForaPartida(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida)

                    newDatasetPartida.team_ultimos_vitorias_home, newDatasetPartida.team_ultimos_empates_home, \
                        newDatasetPartida.team_ultimos_derrotas_home = self.obterUltimosInfoCasaForaPartida(
                            fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                            limitDados=limitHistoricoMedias)

                    (newDatasetPartida.team_max_vitorias_home, newDatasetPartida.team_max_empates_home,
                     newDatasetPartida.team_max_derrotas_home) = (
                        self.obterMaxInfoCasaForaPartida(fixtureTeam=fixtureTeam,arrDatasetPartida=arrDatasetPartida))

                    newDatasetPartida = self.getInfoJogosPassados(id_team=newDatasetPartida.id_team_home,
                                                                  is_home=1, newDataset=newDatasetPartida,
                                                                  arrDatasetPartida=arrDatasetPartida)
                else:
                    # Time jogando fora
                    newDatasetPartida.id_team_away = fixtureTeam.id_team
                    newDatasetPartida.dataset_partida_rotulo.gols_away += fixtureTeam.goals
                    newDatasetPartida.dataset_partida_rotulo.winner = self.getWinnerNormalizadoPartida(
                        is_winner=fixtureTeam.is_winner, is_home=fixtureTeam.is_home)

                    newDatasetPartida.dataset_partida_rotulo.winner_away = self.getWinnerNormalizado(
                        is_winner=fixtureTeam.is_winner)

                    newDatasetPartida.dataset_partida_rotulo.is_winner_away = \
                        (1 if newDatasetPartida.dataset_partida_rotulo.winner_away == 2 else 0)

                    newDatasetPartida.dataset_partida_rotulo.vitoria_away = \
                        1 if newDatasetPartida.dataset_partida_rotulo.winner_away == 2 else 0
                    newDatasetPartida.dataset_partida_rotulo.empate_away = \
                        1 if newDatasetPartida.dataset_partida_rotulo.winner_away == 1 else 0
                    newDatasetPartida.dataset_partida_rotulo.derrota_away = \
                        1 if newDatasetPartida.dataset_partida_rotulo.winner_away == 0 else 0

                    newDatasetPartida.season_pontos_away, newDatasetPartida.season_saldo_gols_away = \
                        self.obterPontuacaoTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                id_season=newDatasetPartida.id_season)

                    newDatasetPartida.team_pontos_away, newDatasetPartida.team_saldo_gols_away = \
                        self.obterPontuacaoTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida)

                    newDatasetPartida.season_media_gols_marcados_away, newDatasetPartida.season_media_gols_sofridos_away = \
                        self.obterMediaGolsTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                id_season=newDatasetPartida.id_season)

                    newDatasetPartida.team_media_gols_marcados_away, newDatasetPartida.team_media_gols_sofridos_away = \
                        self.obterMediaGolsTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                limitDados=limitHistoricoMedias)

                    newDatasetPartida.team_media_resultados_partida_away, \
                        newDatasetPartida.team_melhorando_resultados_partida_away = \
                        self.obterMediaResultadosPartida(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                         limitDados=limitHistoricoMedias)

                    newDatasetPartida.team_empates_casa_away, newDatasetPartida.team_vitorias_casa_away, \
                        newDatasetPartida.team_derrotas_casa_away, newDatasetPartida.team_empates_fora_away, \
                        newDatasetPartida.team_vitorias_fora_away, newDatasetPartida.team_derrotas_fora_away = \
                        self.obterInfoCasaForaPartida(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida)

                    newDatasetPartida.team_ultimos_vitorias_away, newDatasetPartida.team_ultimos_empates_away, \
                        newDatasetPartida.team_ultimos_derrotas_away = self.obterUltimosInfoCasaForaPartida(
                            fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                            limitDados=limitHistoricoMedias)

                    (newDatasetPartida.team_max_vitorias_away, newDatasetPartida.team_max_empates_away,
                     newDatasetPartida.team_max_derrotas_away) = (
                        self.obterMaxInfoCasaForaPartida(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida))

                    newDatasetPartida = self.getInfoJogosPassados(id_team=newDatasetPartida.id_team_away,
                                                                  is_home=0, newDataset=newDatasetPartida,
                                                                  arrDatasetPartida=arrDatasetPartida)

            golsPartida = newDatasetPartida.dataset_partida_rotulo.gols_home + newDatasetPartida.dataset_partida_rotulo.gols_away
            golsHome = newDatasetPartida.dataset_partida_rotulo.gols_home
            golsAway = newDatasetPartida.dataset_partida_rotulo.gols_away
            saldoGolsPartida = newDatasetPartida.dataset_partida_rotulo.gols_home - newDatasetPartida.dataset_partida_rotulo.gols_away
            saldoGolsPartidaHome = newDatasetPartida.dataset_partida_rotulo.gols_home - newDatasetPartida.dataset_partida_rotulo.gols_away
            saldoGolsPartidaAway = newDatasetPartida.dataset_partida_rotulo.gols_away - newDatasetPartida.dataset_partida_rotulo.gols_home

            newDatasetPartida.dataset_partida_rotulo.chance_empate = int(abs(saldoGolsPartida) <= 1)
            newDatasetPartida.dataset_partida_rotulo.chance_empate_home = (
                int((saldoGolsPartidaHome == 1 or saldoGolsPartidaHome == 0)))
            newDatasetPartida.dataset_partida_rotulo.chance_empate_away = (
                int((saldoGolsPartidaAway == 1 or saldoGolsPartidaAway == 0)))

            newDatasetPartida.dataset_partida_rotulo.chance_vitoria = int(abs(saldoGolsPartida) >= 2)
            newDatasetPartida.dataset_partida_rotulo.chance_vitoria_home = int(saldoGolsPartidaHome >= 2)
            newDatasetPartida.dataset_partida_rotulo.chance_vitoria_away = int(saldoGolsPartidaAway >= 2)

            newDatasetPartida.dataset_partida_rotulo.is_superior = int(abs(saldoGolsPartida) >= 2)
            newDatasetPartida.dataset_partida_rotulo.is_superior_home = int(saldoGolsPartidaHome >= 2)
            newDatasetPartida.dataset_partida_rotulo.is_superior_away = int(saldoGolsPartidaAway >= 2)

            if newDatasetPartida.dataset_partida_rotulo.is_superior_home == 1:
                newDatasetPartida.dataset_partida_rotulo.superior = 0
                newDatasetPartida.dataset_partida_rotulo.superior_home = 2
                newDatasetPartida.dataset_partida_rotulo.superior_away = 0
            elif newDatasetPartida.dataset_partida_rotulo.is_superior_away == 1:
                newDatasetPartida.dataset_partida_rotulo.superior = 2
                newDatasetPartida.dataset_partida_rotulo.superior_home = 0
                newDatasetPartida.dataset_partida_rotulo.superior_away = 2
            else:
                newDatasetPartida.dataset_partida_rotulo.superior = 1
                newDatasetPartida.dataset_partida_rotulo.superior_home = 1
                newDatasetPartida.dataset_partida_rotulo.superior_away = 1

            arrDatasetPartida.append(newDatasetPartida)

            if (newDatasetPartida.dataset_partida_rotulo.is_superior_home == 1 and
                    newDatasetPartida.dataset_partida_rotulo.winner_home == 2):
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols = 0
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_home = 0
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_away = 4

            elif (newDatasetPartida.dataset_partida_rotulo.is_superior_home == 0 and
                    newDatasetPartida.dataset_partida_rotulo.winner_home == 2):
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols = 1
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_home = 1
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_away = 3

            elif (newDatasetPartida.dataset_partida_rotulo.is_superior_away == 1 and
                    newDatasetPartida.dataset_partida_rotulo.winner_away == 2):
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols = 4
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_home = 4
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_away = 0

            elif (newDatasetPartida.dataset_partida_rotulo.is_superior_away == 0 and
                  newDatasetPartida.dataset_partida_rotulo.winner_away == 2):
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols = 3
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_home = 3
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_away = 1
            else:
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols = 2
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_home = 2
                newDatasetPartida.dataset_partida_rotulo.winner_by_gols_away = 2

            if golsPartida == 0 or golsPartida == 1:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida = 0
            elif golsPartida == 2 or golsPartida == 3:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida = 1
            else:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida = 2

            if golsHome == 0:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_home = 0
            elif golsHome == 1:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_home = 1
            elif golsHome == 2:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_home = 2
            elif golsHome == 3:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_home = 3
            else:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_home = 4

            if golsAway == 0:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_away = 0
            elif golsAway == 1:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_away = 1
            elif golsAway == 2:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_away = 2
            elif golsAway == 3:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_away = 3
            else:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_away = 4


            if golsAway == 0 or golsAway == 1:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_away = 0
            elif golsAway == 2 or golsAway == 3:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_away = 1
            else:
                newDatasetPartida.dataset_partida_rotulo.qtde_gols_partida_away = 2

            if golsPartida >= 3:
                newDatasetPartida.dataset_partida_rotulo.is_mais_dois_meio = 1
            else:
                newDatasetPartida.dataset_partida_rotulo.is_mais_dois_meio = 0

            if golsHome >= 1:
                newDatasetPartida.dataset_partida_rotulo.is_home_marca = 1

            if golsAway >= 1:
                newDatasetPartida.dataset_partida_rotulo.is_away_marca = 1

            if golsHome >= 1 and golsAway >= 1:
                newDatasetPartida.dataset_partida_rotulo.is_amba_marca = 1

            if golsHome == 0 and golsAway == 0:
                newDatasetPartida.dataset_partida_rotulo.is_none_marca = 1

            if golsPartida <= 0:
                newDatasetPartida.dataset_partida_rotulo.isMenosDeMeio = 0
            elif golsPartida <= 1:
                newDatasetPartida.dataset_partida_rotulo.isMenosDeMeio = 1
            elif golsPartida <= 2:
                newDatasetPartida.dataset_partida_rotulo.isMenosDeMeio = 2
            elif golsPartida <= 3:
                newDatasetPartida.dataset_partida_rotulo.isMenosDeMeio = 3
            else:
                newDatasetPartida.dataset_partida_rotulo.isMenosDeMeio = 4

            if golsPartida > 4:
                newDatasetPartida.dataset_partida_rotulo.isMaisDeMeio = 4
            elif golsPartida > 3:
                newDatasetPartida.dataset_partida_rotulo.isMaisDeMeio = 3
            elif golsPartida > 2:
                newDatasetPartida.dataset_partida_rotulo.isMaisDeMeio = 2
            elif golsPartida > 1:
                newDatasetPartida.dataset_partida_rotulo.isMaisDeMeio = 1
            else:
                newDatasetPartida.dataset_partida_rotulo.isMaisDeMeio = 0

            if golsHome >= 2:
                newDatasetPartida.dataset_partida_rotulo.isMaisDeUmEMeioHome = 1

            if golsAway >= 2:
                newDatasetPartida.dataset_partida_rotulo.isMaisDeUmEMeioAway = 1

            if golsPartida % 2 == 0:
                newDatasetPartida.dataset_partida_rotulo.isPar = 1
            else:
                newDatasetPartida.dataset_partida_rotulo.isPar = 0

        return arrDatasetPartida

    def obterUltimoDatasetPartida(self, arrDatasetPartida: list, id_team: int,
                                  id_season: int = None) -> DatasetPartidaEntrada:
        for datasetPartida in reversed(arrDatasetPartida):
            if id_season is None and (datasetPartida.id_team_home == id_team or datasetPartida.id_team_away == id_team):
                return datasetPartida
            elif id_season == datasetPartida.id_season and \
                    (datasetPartida.id_team_home == id_team or datasetPartida.id_team_away == id_team):

                return datasetPartida

    def obterPontuacaoTeam(self, fixtureTeam: FixtureTeams, arrDatasetPartida: list,
                           id_season: int = None) -> tuple:
        ultimoDatasetPartida: DatasetPartidaEntrada = self.obterUltimoDatasetPartida(
            arrDatasetPartida=arrDatasetPartida,
            id_team=fixtureTeam.id_team,
            id_season=id_season)
        pontos = 0
        saldoGols = 0

        if ultimoDatasetPartida is None:
            return pontos, saldoGols

        if ultimoDatasetPartida.id_team_home == fixtureTeam.id_team:
            if id_season is None:
                pontos = ultimoDatasetPartida.team_pontos_home
                saldoGols = ultimoDatasetPartida.team_saldo_gols_home
            else:
                pontos = ultimoDatasetPartida.season_pontos_home
                saldoGols = ultimoDatasetPartida.season_saldo_gols_home

            if ultimoDatasetPartida.dataset_partida_rotulo.winner_home == 2:
                pontos += 3
            elif ultimoDatasetPartida.dataset_partida_rotulo.winner_home == 1:
                pontos += 1

            saldoGols += ultimoDatasetPartida.dataset_partida_rotulo.gols_home
            saldoGols -= ultimoDatasetPartida.dataset_partida_rotulo.gols_away

        elif ultimoDatasetPartida.id_team_away == fixtureTeam.id_team:
            if id_season is None:
                pontos = ultimoDatasetPartida.team_pontos_away
                saldoGols = ultimoDatasetPartida.team_saldo_gols_away
            else:
                pontos = ultimoDatasetPartida.season_pontos_away
                saldoGols = ultimoDatasetPartida.season_saldo_gols_away

            if ultimoDatasetPartida.dataset_partida_rotulo.winner_away == 2:
                pontos += 3
            elif ultimoDatasetPartida.dataset_partida_rotulo.winner_away == 1:
                pontos += 1

            saldoGols += ultimoDatasetPartida.dataset_partida_rotulo.gols_away
            saldoGols -= ultimoDatasetPartida.dataset_partida_rotulo.gols_home

        return pontos, saldoGols

    def obterMediaGolsTeam(self, fixtureTeam: FixtureTeams, arrDatasetPartida: list,
                           id_season: int = None, limitDados: int = None):

        arrGolsMarcados: list = []
        arrGolsSofridos: list = []
        qtdeDadosEncontados: int = 0
        for datasetPartida in reversed(arrDatasetPartida):
            if id_season is None:
                pass
            elif id_season != datasetPartida.id_season:
                continue

            if datasetPartida.id_team_home == fixtureTeam.id_team:
                qtdeDadosEncontados += 1
                arrGolsMarcados.append(datasetPartida.dataset_partida_rotulo.gols_home)
                arrGolsSofridos.append(datasetPartida.dataset_partida_rotulo.gols_away)
            elif datasetPartida.id_team_away == fixtureTeam.id_team:
                qtdeDadosEncontados += 1
                arrGolsMarcados.append(datasetPartida.dataset_partida_rotulo.gols_away)
                arrGolsSofridos.append(datasetPartida.dataset_partida_rotulo.gols_home)

            if limitDados is not None and qtdeDadosEncontados == limitDados:
                break

        if qtdeDadosEncontados == 0:
            return 0, 0
        else:
            mediaGolsMarcados = sum(arrGolsMarcados) / len(arrGolsMarcados)
            mediaGolsSofridos = sum(arrGolsSofridos) / len(arrGolsSofridos)

            return mediaGolsMarcados, mediaGolsSofridos

    # Média de vitória, derrota e empate
    def obterMediaResultadosPartida(self, fixtureTeam: FixtureTeams, arrDatasetPartida: list,
                                    id_season: int = None, limitDados: int = None):
        ultimoDatasetPartida: DatasetPartidaEntrada = self.obterUltimoDatasetPartida(
            arrDatasetPartida=arrDatasetPartida, id_team=fixtureTeam.id_team, id_season=id_season)

        arrResultadosPartida = []

        for datasetPartida in reversed(arrDatasetPartida):
            if id_season is None:
                pass
            elif id_season != datasetPartida.id_season:
                continue

            if datasetPartida.id_team_home == fixtureTeam.id_team:
                arrResultadosPartida.append(datasetPartida.dataset_partida_rotulo.winner_home)
            elif datasetPartida.id_team_away == fixtureTeam.id_team:
                arrResultadosPartida.append(datasetPartida.dataset_partida_rotulo.winner_away)

            if limitDados is not None and len(arrResultadosPartida) == limitDados:
                break

        if len(arrResultadosPartida) == 0:
            return 0, 0

        mediaResultado = sum(arrResultadosPartida) / len(arrResultadosPartida)

        isMediaMelhorando = False

        if ultimoDatasetPartida.id_team_home == fixtureTeam.id_team:
            isMediaMelhorando = ultimoDatasetPartida.team_media_resultados_partida_home < mediaResultado
        elif ultimoDatasetPartida.id_team_away == fixtureTeam.id_team:
            isMediaMelhorando = ultimoDatasetPartida.team_media_resultados_partida_away < mediaResultado

        return mediaResultado, int(isMediaMelhorando)

    def obterInfoCasaForaPartida(self, fixtureTeam: FixtureTeams, arrDatasetPartida: list,
                                 id_season: int = None, limitDados: int = None):
        ultimoDatasetPartida: DatasetPartidaEntrada = self.obterUltimoDatasetPartida(
            arrDatasetPartida=arrDatasetPartida, id_team=fixtureTeam.id_team, id_season=id_season)

        if ultimoDatasetPartida is None:
            return 0, 0, 0, 0, 0, 0

        empateCasa = ultimoDatasetPartida.team_empates_casa_home if ultimoDatasetPartida.id_team_home == fixtureTeam.id_team else \
            ultimoDatasetPartida.team_empates_casa_away
        vitoriaCasa = ultimoDatasetPartida.team_vitorias_casa_home if ultimoDatasetPartida.id_team_home == fixtureTeam.id_team else \
            ultimoDatasetPartida.team_vitorias_casa_away
        derrotaCasa = ultimoDatasetPartida.team_derrotas_casa_home if ultimoDatasetPartida.id_team_home == fixtureTeam.id_team else \
            ultimoDatasetPartida.team_derrotas_casa_away

        empateFora = ultimoDatasetPartida.team_empates_fora_home if ultimoDatasetPartida.id_team_home == fixtureTeam.id_team else \
            ultimoDatasetPartida.team_empates_fora_away
        vitoriaFora = ultimoDatasetPartida.team_vitorias_fora_home if ultimoDatasetPartida.id_team_home == fixtureTeam.id_team else \
            ultimoDatasetPartida.team_vitorias_fora_away
        derrotaFora = ultimoDatasetPartida.team_derrotas_fora_home if ultimoDatasetPartida.id_team_home == fixtureTeam.id_team else \
            ultimoDatasetPartida.team_derrotas_fora_away

        if ultimoDatasetPartida.id_team_home == fixtureTeam.id_team:
            if ultimoDatasetPartida.dataset_partida_rotulo.winner_home == 1:
                empateCasa += 1
            elif ultimoDatasetPartida.dataset_partida_rotulo.winner_home == 2:
                vitoriaCasa += 1
            elif ultimoDatasetPartida.dataset_partida_rotulo.winner_home == 0:
                derrotaCasa += 1
        elif ultimoDatasetPartida.id_team_away == fixtureTeam.id_team:
            if ultimoDatasetPartida.dataset_partida_rotulo.winner_away == 1:
                empateFora += 1
            elif ultimoDatasetPartida.dataset_partida_rotulo.winner_away == 2:
                vitoriaFora += 1
            elif ultimoDatasetPartida.dataset_partida_rotulo.winner_away == 0:
                derrotaFora += 1

        return empateCasa, vitoriaCasa, derrotaCasa, empateFora, vitoriaFora, derrotaFora

    def obterUltimosInfoCasaForaPartida(self, fixtureTeam: FixtureTeams, arrDatasetPartida: list,
                                        id_season: int = None, limitDados: int = None):
        ultimoDatasetPartida: DatasetPartidaEntrada = self.obterUltimoDatasetPartida(
            arrDatasetPartida=arrDatasetPartida, id_team=fixtureTeam.id_team, id_season=id_season)

        arrResultadosPartida = []

        for datasetPartida in reversed(arrDatasetPartida):
            if id_season is None:
                pass
            elif id_season != datasetPartida.id_season:
                continue

            if datasetPartida.id_team_home == fixtureTeam.id_team:
                arrResultadosPartida.append(datasetPartida.dataset_partida_rotulo.winner_home)
            elif datasetPartida.id_team_away == fixtureTeam.id_team:
                arrResultadosPartida.append(datasetPartida.dataset_partida_rotulo.winner_away)

            if limitDados is not None and len(arrResultadosPartida) == limitDados:
                break

        if len(arrResultadosPartida) == 0:
            return 0, 0, 0

        isHome = ultimoDatasetPartida.id_team_home == fixtureTeam.id_team
        vitorias = 0
        derrotas = 0
        empates = 0

        for resul in arrResultadosPartida:
            if resul == 1:
                empates += 1
            elif resul == 2:
                vitorias += 1
            elif resul == 0:
                derrotas += 1

        return vitorias, empates, derrotas

    def obterMaxInfoCasaForaPartida(self, fixtureTeam: FixtureTeams, arrDatasetPartida: list,
                                    id_season: int = None, limitDados: int = None):

        ultimoDatasetPartida: DatasetPartidaEntrada = self.obterUltimoDatasetPartida(
            arrDatasetPartida=arrDatasetPartida, id_team=fixtureTeam.id_team, id_season=id_season)

        vitorias, empates, derrotas = 0, 0, 0

        if ultimoDatasetPartida is None:
            return 0, 0, 0

        nEmpates = 0
        nVitorias = 0
        nDerrotas = 0

        for datasetPartida in reversed(arrDatasetPartida):
            if datasetPartida.id_team_home == fixtureTeam.id_team:
                if datasetPartida.dataset_partida_rotulo.winner_home == 2:
                    nVitorias += 1
                    empates = empates if empates >= nEmpates else nEmpates
                    nEmpates = 0
                    derrotas = derrotas if derrotas >= nDerrotas else nDerrotas
                    nDerrotas = 0

                elif datasetPartida.dataset_partida_rotulo.winner_home == 1:
                    nEmpates += 1
                    vitorias = vitorias if vitorias >= nVitorias else nVitorias
                    nVitorias = 0
                    derrotas = derrotas if derrotas >= nDerrotas else nDerrotas
                    nDerrotas = 0

                elif datasetPartida.dataset_partida_rotulo.winner_home == 0:
                    nDerrotas += 1
                    vitorias = vitorias if vitorias >= nVitorias else nVitorias
                    nVitorias = 0
                    empates = empates if empates >= nEmpates else nEmpates
                    nEmpates = 0

            elif datasetPartida.id_team_away == fixtureTeam.id_team:
                if datasetPartida.dataset_partida_rotulo.winner_away == 2:
                    nVitorias += 1
                    empates = empates if empates >= nEmpates else nEmpates
                    nEmpates = 0
                    derrotas = derrotas if derrotas >= nDerrotas else nDerrotas
                    nDerrotas = 0

                elif datasetPartida.dataset_partida_rotulo.winner_away == 1:
                    nEmpates += 1
                    vitorias = vitorias if vitorias >= nVitorias else nVitorias
                    nVitorias = 0
                    derrotas = derrotas if derrotas >= nDerrotas else nDerrotas
                    nDerrotas = 0

                elif datasetPartida.dataset_partida_rotulo.winner_away == 0:
                    nDerrotas += 1
                    vitorias = vitorias if vitorias >= nVitorias else nVitorias
                    nVitorias = 0
                    empates = empates if empates >= nEmpates else nEmpates
                    nEmpates = 0

        vitorias = vitorias if vitorias >= nVitorias else nVitorias
        empates = empates if empates >= nEmpates else nEmpates
        derrotas = derrotas if derrotas >= nDerrotas else nDerrotas
        return vitorias, empates, derrotas

    def getWinnerNormalizadoPartida(self, is_winner: int, is_home: int) -> int:
        if is_winner is None:
            return 1

        if is_home == 1:
            return 0 if is_winner == 1 else 2
        elif is_home == 0:
            return 2 if is_winner == 1 else 0

    def getWinnerNormalizado(self, is_winner: int) -> int:
        if is_winner is None:
            return 1
        elif is_winner == 0:
            return 0
        elif is_winner == 1:
            return 2

    def getInfoJogosPassados(self, id_team: int, is_home: int, newDataset: DatasetPartidaEntrada,
                             arrDatasetPartida: list):
        lastJogosTeam = self.obterUltimoDatasetPartida(arrDatasetPartida=arrDatasetPartida, id_team=id_team)

        if lastJogosTeam is None or len(arrDatasetPartida) == 0:
            return newDataset

        lastTeamHome = lastJogosTeam.id_team_home == id_team

        team_jogo_gols_marcados = lastJogosTeam.dataset_partida_rotulo.gols_home if lastTeamHome \
            else lastJogosTeam.dataset_partida_rotulo.gols_away

        team_jogo_gols_sofridos = lastJogosTeam.dataset_partida_rotulo.gols_away if lastTeamHome \
            else lastJogosTeam.dataset_partida_rotulo.gols_home

        team_jogo_isPar = lastJogosTeam.dataset_partida_rotulo.isPar

        team_jogo_winner_by_gols = lastJogosTeam.dataset_partida_rotulo.winner_by_gols_home if lastTeamHome \
            else lastJogosTeam.dataset_partida_rotulo.winner_by_gols_away

        team_jogo_isSuperior = lastJogosTeam.dataset_partida_rotulo.is_superior_home if lastTeamHome \
            else lastJogosTeam.dataset_partida_rotulo.is_superior_away

        team_jogo_isMenosDeMeio = lastJogosTeam.dataset_partida_rotulo.isMenosDeMeio

        team_jogo_winner = lastJogosTeam.dataset_partida_rotulo.winner_home if lastJogosTeam \
            else lastJogosTeam.dataset_partida_rotulo.winner_away

        if is_home == 1:
            newDataset.team_jogo_gols_marcados_home = team_jogo_gols_marcados
            newDataset.team_jogo_gols_sofridos_home = team_jogo_gols_sofridos
            newDataset.team_jogo_isPar_home = team_jogo_isPar
            newDataset.team_jogo_winner_by_gols_home = team_jogo_winner_by_gols
            newDataset.team_jogo_isSuperior_home = team_jogo_isSuperior
            newDataset.team_jogo_isMenosDeMeio_home = team_jogo_isMenosDeMeio
            newDataset.team_jogo_winnerHome = team_jogo_winner
        else:
            newDataset.team_jogo_gols_marcados_away = team_jogo_gols_marcados
            newDataset.team_jogo_gols_sofridos_away = team_jogo_gols_sofridos
            newDataset.team_jogo_isPar_away = team_jogo_isPar
            newDataset.team_jogo_winner_by_gols_away = team_jogo_winner_by_gols
            newDataset.team_jogo_isSuperior_away = team_jogo_isSuperior
            newDataset.team_jogo_isMenosDeMeio_away = team_jogo_isMenosDeMeio
            newDataset.team_jogo_winnerAway = team_jogo_winner

        return newDataset

