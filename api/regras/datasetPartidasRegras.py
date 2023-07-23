import math

from api.models.fixturesModel import Fixture
from api.models.fixturesTeamsModel import FixtureTeams

from api.regras.fixturesRegras import FixturesRegras
from api.regras.iaUteisRegras import IAUteisRegras

class DatasetPartidaRotulo:
    def __init__(self):
        #winner deve ser 0 vitoria casa, 1 empate, 2 vitoria fora
        self.winner: int = None
        self.chance_empate: int = None
        self.chance_vitoria: int = None
        self.winner_home: int = None
        self.winner_away: int = None
        self.gols_home: int = 0
        self.gols_away: int = 0


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

        #media deve ficar em um range de 0 a 2 sendo 0 derrota 1 empate e 2 vitoria
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

        # media deve ficar em um range de 0 a 2 sendo 0 derrota 1 empate e 2 vitoria
        self.team_media_resultados_partida_away: float = 0.0
        self.team_melhorando_resultados_partida_away: int = None

    def obterDadoEntradaEmArray(self, arridsTeam: list[int] = None, id_jogo: int  = 0, isAntiga: bool = False):
        isHomePontuacaoMaior = 1 if self.season_pontos_home >= self.season_pontos_away else 0
        if isAntiga:
            if self.id_team_home in arridsTeam and self.id_team_away in arridsTeam:
                arrDados = [[id_jogo, isHomePontuacaoMaior, self.id_team_home, self.id_season, self.season_pontos_home, self.team_saldo_gols_home,
                            self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                            self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home, 1,

                            self.id_team_away, self.id_season, self.season_pontos_away, self.team_saldo_gols_away,
                            self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                            self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away],

                            [id_jogo, isHomePontuacaoMaior, self.id_team_away, self.id_season, self.season_pontos_away, self.team_saldo_gols_away,
                             self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                             self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away, 0,

                             self.id_team_home, self.id_season, self.season_pontos_home, self.team_saldo_gols_home,
                             self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                             self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home]]
            elif self.id_team_away in arridsTeam:
                arrDados = [[id_jogo, isHomePontuacaoMaior, self.id_team_away, self.id_season, self.season_pontos_away, self.team_saldo_gols_away,
                            self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                            self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away, 0,

                            self.id_team_home, self.id_season, self.season_pontos_home, self.team_saldo_gols_home,
                            self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                            self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home]]
            else:
                arrDados = [[id_jogo, isHomePontuacaoMaior, self.id_team_home, self.id_season, self.season_pontos_home, self.team_saldo_gols_home,
                             self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                             self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home, 1,

                             self.id_team_away, self.id_season, self.season_pontos_away, self.team_saldo_gols_away,
                             self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                             self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away]]
        else:
            arrDados = [[id_jogo, isHomePontuacaoMaior, self.id_team_home, self.id_season, self.season_pontos_home,
                         self.team_saldo_gols_home,
                         self.team_media_gols_marcados_home, self.team_media_gols_sofridos_home,
                         self.team_media_resultados_partida_home, self.team_melhorando_resultados_partida_home,

                         self.id_team_away, self.id_season, self.season_pontos_away, self.team_saldo_gols_away,
                         self.team_media_gols_marcados_away, self.team_media_gols_sofridos_away,
                         self.team_media_resultados_partida_away, self.team_melhorando_resultados_partida_away]]

        return arrDados

    def obterDadoRotulosEmArray(self, arridsTeam: list[int] = None, isAntiga: bool = False):
        #cada array interno representa uma camada de saida.
        if isAntiga:
            if self.id_team_home in arridsTeam and self.id_team_away in arridsTeam:
                arrDados = [[[self.dataset_partida_rotulo.winner_home]],
                            [[self.dataset_partida_rotulo.winner_away]]]
            elif self.id_team_away in arridsTeam:
                arrDados = [[[self.dataset_partida_rotulo.winner_away]]]
            else:
                arrDados = [[[self.dataset_partida_rotulo.winner_home]]]
        else:
            arrDados = [[[self.dataset_partida_rotulo.winner]]]

        return arrDados

class DatasetPartidasRegras:
    def obter(self, arrIdsTeam: list[int], limitHistoricoMedias: int = 5, isNormalizarSaidaEmClasse: bool = True,
              isFiltrarTeams: bool = True):
        iaUteisRegras = IAUteisRegras()

        arrDatasetEntradaEmArray = []
        arrDatasetRotuloEmArray = []
        arrIndexDadosRemover = []
        arrDatasetPartida = self.obterInfosDatabase(arrIdsTeam=arrIdsTeam, limitHistoricoMedias=limitHistoricoMedias)

        indexDadosPrever = -1
        id_jogo = 0
        for datasetPartida in arrDatasetPartida:
            id_jogo += 1
            for retEntrada in datasetPartida.obterDadoEntradaEmArray(arridsTeam=arrIdsTeam, id_jogo=id_jogo):
                indexDadosPrever += 1

                if datasetPartida.is_prever == 1:
                    arrIndexDadosRemover.append(indexDadosPrever)

                arrDatasetEntradaEmArray.append(retEntrada)

            for retRotulo in datasetPartida.obterDadoRotulosEmArray(arridsTeam=arrIdsTeam):
                arrDatasetRotuloEmArray.append(retRotulo)

        arrDatasetEntradaEmArrayNormalizado, max_value_entrada, min_value_entrada = \
            iaUteisRegras.normalizar_dataset(arrDatasetEntradaEmArray)

        arrDatasetRotuloEmArrayNormalizado, max_value_rotulo, min_value_rotulo = \
            iaUteisRegras.normalizar_dataset(arrDatasetRotuloEmArray)

        arrDatasetPreverEmArrayNormalizado = []

        for indexRemover in reversed(arrIndexDadosRemover):
            elementoPrever = arrDatasetEntradaEmArrayNormalizado.pop(indexRemover)
            rotuloRemovido = arrDatasetRotuloEmArrayNormalizado.pop(indexRemover)
            #elementoPrever.pop(12)
            #elementoPrever.pop(0)
            arrDatasetPreverEmArrayNormalizado.append(elementoPrever)

        if isNormalizarSaidaEmClasse:
            arrDatasetRotuloEmArrayNormalizado = self.normalizarSaidaEmClasses(arrDatasetRotuloEmArrayNormalizado)

        arrAAA = []
        arrBBB = []
        qtdeDados = 500
        if not isFiltrarTeams:
            return arrDatasetEntradaEmArrayNormalizado[-qtdeDados:], \
                arrDatasetRotuloEmArrayNormalizado[-qtdeDados:], \
                arrDatasetPreverEmArrayNormalizado

        for indexDadoEntrada in range(len(arrDatasetEntradaEmArrayNormalizado)):
            isEncontrou = False
            for indexValue in range(len(arrDatasetEntradaEmArrayNormalizado[indexDadoEntrada])):
                if indexValue == 2 or indexValue == 10:
                    id_team = iaUteisRegras.desnormalizarValue(
                        value_normalizado=arrDatasetEntradaEmArrayNormalizado[indexDadoEntrada][indexValue],
                        max_value=max_value_entrada[indexValue], min_value=min_value_entrada[indexValue])

                    id_team = round(id_team)

                    if id_team in arrIdsTeam:
                        isEncontrou = True
                        break
            if isEncontrou:
                #arrDatasetEntradaEmArrayNormalizado[indexDadoEntrada].pop(12)
                #arrDatasetEntradaEmArrayNormalizado[indexDadoEntrada].pop(0)
                arrAAA.append(arrDatasetEntradaEmArrayNormalizado[indexDadoEntrada])
                arrBBB.append(arrDatasetRotuloEmArrayNormalizado[indexDadoEntrada])


        return arrAAA[-50:], arrBBB[-50:], arrDatasetPreverEmArrayNormalizado


    def normalizarSaidaEmClasses(self, arrRotulos: list[list[list]]):
        arrIndexCamadaValuesSaida = [[] for i in range(len(arrRotulos[0]))]

        for indexCamadaSaida in range(len(arrIndexCamadaValuesSaida)):
            for i in range(len(arrRotulos[0][indexCamadaSaida])):
                arrIndexCamadaValuesSaida[indexCamadaSaida].append([])

        for dadoRotulo in arrRotulos:
            for indexCamada in range(len(arrIndexCamadaValuesSaida)):
                for indexValue in range(len(arrIndexCamadaValuesSaida[indexCamada])):
                    if dadoRotulo[indexCamada][indexValue] not in arrIndexCamadaValuesSaida[indexCamada][indexValue]:
                        arrIndexCamadaValuesSaida[indexCamada][indexValue].append(dadoRotulo[indexCamada][indexValue])
                        arrIndexCamadaValuesSaida[indexCamada][indexValue] = sorted(arrIndexCamadaValuesSaida[indexCamada][indexValue])
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
                        indexValueIgual = arrIndexCamadaValuesSaida[indexCamada][indexValue].index(dadoRotulo[indexCamada][indexValue])
                        arrDadosValueNormalizado[indexValueIgual] = 1

                    for valueClasses in arrDadosValueNormalizado:
                        arrDadosClasseNormalizado.append(valueClasses)
                arrCamadaNormalizada.append(arrDadosClasseNormalizado)
            rotulosNormalizadosEmClasse.append(arrCamadaNormalizada)
        return rotulosNormalizadosEmClasse


    def obterInfosDatabase(self, arrIdsTeam: list[int], limitHistoricoMedias):
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

                    newDatasetPartida.season_pontos_home, newDatasetPartida.season_saldo_gols_home = \
                        self.obterPontuacaoTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                id_season=newDatasetPartida.id_season)

                    newDatasetPartida.team_pontos_home, newDatasetPartida.team_saldo_gols_home = \
                        self.obterPontuacaoTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida)

                    newDatasetPartida.season_media_gols_marcados_home, newDatasetPartida.season_media_gols_sofridos_home = \
                        self.obterMediaGolsTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                limitDados=limitHistoricoMedias)

                    newDatasetPartida.team_media_gols_marcados_home, newDatasetPartida.team_media_gols_sofridos_home = \
                        self.obterMediaGolsTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida)

                    newDatasetPartida.team_media_resultados_partida_home, \
                        newDatasetPartida.team_melhorando_resultados_partida_home = \
                            self.obterMediaResultadosPartida(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                             limitDados=limitHistoricoMedias)
                else:
                    # Time jogando fora
                    newDatasetPartida.id_team_away = fixtureTeam.id_team
                    newDatasetPartida.dataset_partida_rotulo.gols_away += fixtureTeam.goals
                    newDatasetPartida.dataset_partida_rotulo.winner = self.getWinnerNormalizadoPartida(
                        is_winner=fixtureTeam.is_winner, is_home=fixtureTeam.is_home)

                    newDatasetPartida.dataset_partida_rotulo.winner_away = self.getWinnerNormalizado(
                        is_winner=fixtureTeam.is_winner)

                    newDatasetPartida.season_pontos_away, newDatasetPartida.season_saldo_gols_away = \
                        self.obterPontuacaoTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                id_season=newDatasetPartida.id_season)

                    newDatasetPartida.team_pontos_away, newDatasetPartida.team_saldo_gols_away = \
                        self.obterPontuacaoTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida)

                    newDatasetPartida.season_media_gols_marcados_away, newDatasetPartida.season_media_gols_sofridos_away = \
                        self.obterMediaGolsTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                limitDados=limitHistoricoMedias)

                    newDatasetPartida.team_media_gols_marcados_away, newDatasetPartida.team_media_gols_sofridos_away = \
                        self.obterMediaGolsTeam(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida)

                    newDatasetPartida.team_media_resultados_partida_away, \
                        newDatasetPartida.team_melhorando_resultados_partida_away = \
                            self.obterMediaResultadosPartida(fixtureTeam=fixtureTeam, arrDatasetPartida=arrDatasetPartida,
                                                             limitDados=limitHistoricoMedias)

            saldoGolsPartida = newDatasetPartida.dataset_partida_rotulo.gols_home - newDatasetPartida.dataset_partida_rotulo.gols_away
            newDatasetPartida.dataset_partida_rotulo.chance_empate = int(abs(saldoGolsPartida) <= 1)
            newDatasetPartida.dataset_partida_rotulo.chance_vitoria = int(abs(saldoGolsPartida) >= 1)
            arrDatasetPartida.append(newDatasetPartida)
        return arrDatasetPartida


    def obterUltimoDatasetPartida(self, arrDatasetPartida: list[DatasetPartidaEntrada], id_team: int, id_season: int = None) -> DatasetPartidaEntrada:
        for datasetPartida in reversed(arrDatasetPartida):
            if id_season is None and (datasetPartida.id_team_home == id_team or datasetPartida.id_team_away == id_team):
                return datasetPartida
            elif id_season == datasetPartida.id_season and \
                (datasetPartida.id_team_home == id_team or datasetPartida.id_team_away == id_team):

                return datasetPartida

    def obterPontuacaoTeam(self, fixtureTeam: FixtureTeams, arrDatasetPartida: list[DatasetPartidaEntrada],
                           id_season: int = None) -> tuple[int, int]:
        ultimoDatasetPartida: DatasetPartidaEntrada = self.obterUltimoDatasetPartida(arrDatasetPartida=arrDatasetPartida,
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


    def obterMediaGolsTeam(self, fixtureTeam: FixtureTeams, arrDatasetPartida: list[DatasetPartidaEntrada],
                           id_season: int = None, limitDados: int = None):

        arrGolsMarcados: list[int] = []
        arrGolsSofridos: list[int] = []
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

    #Média de vitória, derrota e empate
    def obterMediaResultadosPartida(self, fixtureTeam: FixtureTeams, arrDatasetPartida: list[DatasetPartidaEntrada],
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

    def getWinnerNormalizadoPartida(self, is_winner: int, is_home: int) -> int:
        if is_winner is None:
            return 1

        if is_home == 1:
            return  0 if is_winner == 1 else 2
        elif is_home == 0:
            return 2 if is_winner == 1 else 0

    def getWinnerNormalizado(self, is_winner: int) -> int:
        if is_winner is None:
            return 1
        elif is_winner == 0:
            return 0
        elif is_winner == 1:
            return 2