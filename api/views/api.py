import numpy
from datetime import datetime, timedelta
from json import loads, JSONDecoder, JSONEncoder, dumps
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from operator import itemgetter

from api.regras.countriesRegras import CountriesRegras
from api.regras.leaguesSeasonsRegras import LeaguesRegras, SeasonsRegras
from api.regras.teamsRegras import TeamsRegras
from api.regras.uteisRegras import UteisRegras
from api.regras.statisticsRegras import StatisticsRegras
from api.regras.iaRNNRegras import RNN
from api.regras.iaDBNRegras import DBN
from api.regras.iaRegras import IARegras
from api.regras.tabelaJogosRegras import TabelaJogosRegras
from api.regras.tabelaPontuacaoRegras import TabelaPontuacaoRegras
from api.regras.fixturesRegras import FixturesRegras
# Create your views here.

from api.models.countriesModel import Country
from api.models.leaguesModel import League
from api.models.teamsModel import Team


def obterCountries(request):
    uteisRegras = UteisRegras()
    contriesRegras = CountriesRegras()

    idCountry = request.GET.get("id_country")
    arrCountries = contriesRegras.obter(id=idCountry)
    arrCountries = uteisRegras.normalizarDadosForView(arrDados=arrCountries)
    return JsonResponse({"response": arrCountries}, safe=False)


def obterLeagues(request):
    uteisRegras = UteisRegras()
    leaguesRegras = LeaguesRegras()
    idCountry = request.GET.get("id_country")

    if idCountry is None:
        raise "É necessário para o prametro id_country"

    arrLeagues = leaguesRegras.obter(idCountry=idCountry)
    arrLeagues = uteisRegras.normalizarDadosForView(arrDados=arrLeagues)
    return JsonResponse({"response": arrLeagues}, safe=False)


def obterSeasons(request):
    uteisRegras = UteisRegras()
    seasonsRegras = SeasonsRegras()
    idLeague = request.GET.get("id_league")

    if idLeague is None:
        raise "É necessário passar o prametro id_league"

    seasonsRegras.leaguesRegras.leaguesModel.atualizarSeasonsByIdLeague(id_league=idLeague)
    arrSeasons = seasonsRegras.obter(idLeague=idLeague)
    arrSeasons = uteisRegras.normalizarDadosForView(arrDados=arrSeasons)
    return JsonResponse({"response": arrSeasons}, safe=False)


def atualizarSeasonsTeamsFixtures(request):
    teamsRegras = TeamsRegras()
    fixturesRegras = FixturesRegras()
    seasonsRegras = SeasonsRegras()
    idSeason = request.GET.get("id_season")

    if idSeason is None:
        raise "É necessário passar o prametro id_season"


    teamsRegras.teamsModel.atualizarTeamsByLeagueSeason(id_season=int(idSeason), isAtualizarLastModification=False)
    fixturesRegras.fixturesModel.atualizarFixturesByIdSeason(id_season=int(idSeason))
    return JsonResponse({"response": "Team da Season atualizados"}, safe=False)


def searchTeams(request):
    uteisRegras = UteisRegras()
    teamsRegras = TeamsRegras()

    name = request.GET.get("name")
    idSeason = request.GET.get("id_season")

    if idSeason is None:
        raise "É necessário passar o prametro id_season"

    arrTeams = teamsRegras.obter(name=name, id_season=idSeason)
    arrTeams = uteisRegras.normalizarDadosForView(arrDados=arrTeams)
    return JsonResponse({"response": arrTeams}, safe=False)

def obterStatistcsTeamsPlay(request):
    iaRegras = IARegras()
    rnnPartida = RNN(1, [1], [1])

    rbm = DBN(25, 25, 0.01)
    uteisRegras = UteisRegras()
    statisticsRegras = StatisticsRegras()
    idSeason = request.GET.get("id_season")
    idTeamHome = request.GET.get("id_team_home")
    idTeamAway = request.GET.get("id_team_away")

    if idSeason is None and idTeamHome is None:
        raise "É necessário passar o prametro id_season ou id_team"

    isTreinarPartidas = True

    print("############## new Request #######################")
    idSeason = int(idSeason)
    idTeamHome = int(idTeamHome)
    idTeamAway = int(idTeamAway) if idTeamAway is not None else None

    statisticsRegras.fixturesRegras.fixturesModel.atualizarFixturesByidTeam(id_team=idTeamHome);

    if idTeamAway is not None:
        statisticsRegras.fixturesRegras.fixturesModel.atualizarFixturesByidTeam(id_team=idTeamAway);

    print("######### Treinando Partida ##########")
    try:
        arrTeamsPlayPartida = statisticsRegras.obterAllFixturesByIdTeams(idTeamPrincipal=idTeamHome,
                                                                         idTeamAdversario=idTeamAway,
                                                                         id_season=idSeason)
    except:
        return JsonResponse({"erro": "Não consegui obter a relação entre esses dois times,"
                                     " não se preocupe até o dia do jogo terei as informações."}, safe=False)

    datasetTeamsPlayPartida, qtdeAllDados, qtdeDadosHome, qtdeDadosAway = statisticsRegras.normalizarDadosTeamsPlayDataset(arrTeamsPlays=arrTeamsPlayPartida,
                                                                       arrIdsTeamPrever=[idTeamHome, idTeamAway],
                                                                       qtdeDados=30, isFiltrarTeams=True)
    arrPrevTreino = []
    arrPrevPartida, loss = rnnPartida.treinarRNN(datasetRNN=datasetTeamsPlayPartida)
    data_jogo_prevista = None

    for teamsPlay in arrTeamsPlayPartida:
        if teamsPlay.is_prever == 1:
            data_jogo_prevista = (teamsPlay.data_fixture - timedelta(hours=3.0)).strftime("%Y-%m-%d %H:%M:%S")
            break

    dictPrevPartida = {
        "v_ia": "0.35.1",
        "erro": loss,
        "qtde_dados_home": qtdeDadosHome,
        "qtde_dados_away": qtdeDadosAway,
        "data_jogo_previsto": data_jogo_prevista
    }

    for i in range(len(datasetTeamsPlayPartida.arr_name_values_saida)):
        name_chave_prob = datasetTeamsPlayPartida.arr_name_values_saida[i]
        dictPrevPartida[name_chave_prob] = {}
        dictPrevPartida[name_chave_prob] = {
            "vitoria": arrPrevPartida[0][i][2],
            "empate": arrPrevPartida[0][i][1],
            "derrota": arrPrevPartida[0][i][0]
        }

    arrPrevTreino.append(datasetTeamsPlayPartida.arr_name_values_saida)
    arrPrevTreino.append(str("Previsoes Partida: \n" + ",".join(list(map(str, arrPrevPartida)))))


    print("######### Previsões ##########")
    name_team_home = statisticsRegras.teamsRegras.teamsModel.obterByColumnsID(arrDados=[idTeamHome])[0].name
    name_team_away = statisticsRegras.teamsRegras.teamsModel.obterByColumnsID(arrDados=[idTeamAway])[0].name
    print("Team home: ", name_team_home, "Team away: ", name_team_away)
    for prev in arrPrevTreino:
        print(prev)

    return JsonResponse({"response": dictPrevPartida}, safe=False)

def obterPrevisaoTeam(request):
    iaRegras = IARegras()
    rnnPartida = RNN(1, [1], [1])
    rbm = DBN(25, 25, 0.01)
    uteisRegras = UteisRegras()
    statisticsRegras = StatisticsRegras()
    idSeason = request.GET.get("id_season")
    idTeam = request.GET.get("id_team")

    if idSeason is None and idTeam is None:
        raise "É necessário passar o prametro id_season ou id_team"

    print("############## new Request #######################")
    idSeason = int(idSeason)
    idTeam = int(idTeam)

    statisticsRegras.fixturesRegras.fixturesModel.atualizarFixturesByidTeam(id_team=idTeam);

    print("######### Treinando Team ##########")
    try:
        arrTeamsPlay = statisticsRegras.obterAllFixturesByIdTeams(idTeamPrincipal=idTeam, id_season=idSeason)
    except:
        return JsonResponse({"erro": "Não consegui obter a relação entre esses dois times,"
                                     " não se preocupe até o dia do jogo terei as informações."}, safe=False)

    datasetTeamsPlay, qtdeAllDados, qtdeDadosHome, qtdeDadosAway = statisticsRegras.normalizarDadosTeamsPlayDataset(arrTeamsPlays=arrTeamsPlay,
                                                                                        arrIdsTeamPrever=[idTeam],
                                                                                        qtdeDados=25,
                                                                                        isFiltrarTeams=True)
    arrPrevTreino = []
    arrPrevPartida, loss = rnnPartida.treinarRNN(datasetRNN=datasetTeamsPlay, nEpocas=750)
    data_jogo_prevista = None

    for teamsPlay in arrTeamsPlay:
        if teamsPlay.is_prever == 1:
            data_jogo_prevista = (teamsPlay.data_fixture - timedelta(hours=3.0)).strftime("%Y-%m-%d %H:%M:%S")
            break

    dictPrevPartida = {
        "v_ia": "0.35.1",
        "erro": loss,
        "qtde_dados_home": qtdeDadosHome,
        "qtde_dados_away": qtdeDadosAway,
        "data_jogo_previsto": data_jogo_prevista
    }

    for i in range(len(datasetTeamsPlay.arr_name_values_saida)):
        name_chave_prob = datasetTeamsPlay.arr_name_values_saida[i]
        dictPrevPartida[name_chave_prob] = {}
        dictPrevPartida[name_chave_prob] = {
            "vitoria": arrPrevPartida[0][i][2],
            "empate": arrPrevPartida[0][i][1],
            "derrota": arrPrevPartida[0][i][0]
        }

    arrPrevTreino.append(datasetTeamsPlay.arr_name_values_saida)
    arrPrevTreino.append(str("Previsoes Partida: \n" + ",".join(list(map(str, arrPrevPartida)))))


    print("######### Previsões ##########")
    name_team_home = statisticsRegras.teamsRegras.teamsModel.obterByColumnsID(arrDados=[idTeam])[0].name
    print("Team home: ", name_team_home)
    for prev in arrPrevTreino:
        print(prev)

    return JsonResponse({"response": dictPrevPartida}, safe=False)

def obterTabelaPontuacao(request):
    tabelaPontucaoRegras = TabelaPontuacaoRegras()
    uteisRegras = UteisRegras()

    idSeason = request.GET.get("id_season")

    if idSeason is None:
        raise "Sem id_season como parametro para obter tabela pontuacao"

    tabelaPontuacao = tabelaPontucaoRegras.obterTabelaPontucao(id_season=idSeason)
    tabelaPontuacaoNormalizada = uteisRegras.normalizarDadosForView(arrDados=[tabelaPontuacao])[0]
    return JsonResponse({"response": tabelaPontuacaoNormalizada}, safe=True)

def obterTabelaJogos(request):
    tabelaJogosRegras = TabelaJogosRegras()
    uteisRegras = UteisRegras()

    idSeason = request.GET.get("id_season")

    if idSeason is None:
        raise "Sem id_season como parametro para obter tabela pontuacao"

    tabelajogos = tabelaJogosRegras.obterTabelaJogos(id_season=idSeason)
    tabelaJogosNormalizada = uteisRegras.normalizarDadosForView(arrDados=[tabelajogos])[0]
    return JsonResponse({"response": tabelaJogosNormalizada}, safe=True)