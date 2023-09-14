import numpy
import random
from copy import deepcopy
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
from api.regras.iaUteisRegras import IAUteisRegras
from api.regras.tabelaJogosRegras import TabelaJogosRegras
from api.regras.tabelaPontuacaoRegras import TabelaPontuacaoRegras
from api.regras.fixturesRegras import FixturesRegras
from api.regras.datasetFixturesRegras import DatasetFixtureRegras
from api.regras.iaAprendizado import RedeLTSM, ModelPrevisao

# Create your views here.

from api.models.countriesModel import Country
from api.models.leaguesModel import League
from api.models.teamsModel import Team

from api.models.model import Database

database = Database()

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

    arrSeasons = seasonsRegras.obter(idLeague=idLeague)
    arrSeasons = uteisRegras.normalizarDadosForView(arrDados=arrSeasons)
    return JsonResponse({"response": arrSeasons}, safe=False)


def searchTeams(request):
    uteisRegras = UteisRegras()
    teamsRegras = TeamsRegras()

    name = request.GET.get("name")
    idSeason = request.GET.get("id_season")

    if idSeason is None:
        raise "É necessário passar o prametro id_season"

    teamsRegras.teamsModel.atualizarDados(id_season=idSeason)

    arrTeams = teamsRegras.obter(name=name, id_season=idSeason)
    arrTeams = uteisRegras.normalizarDadosForView(arrDados=arrTeams)
    return JsonResponse({"response": arrTeams}, safe=False)


def obterPrevisaoPartida(request):
    iaRegras = IAUteisRegras()
    iaLTSM = RedeLTSM()
    uteisRegras = UteisRegras()
    statisticsRegras = StatisticsRegras()
    fixturesRegras = FixturesRegras()
    idSeason = request.GET.get("id_season")
    idTeamHome = request.GET.get("id_team_home")
    idTeamAway = request.GET.get("id_team_away")

    if idSeason is None and idTeamHome is None:
        raise "É necessário passar o prametro id_season ou id_team"

    idSeason = int(idSeason)
    idTeamHome = int(idTeamHome)
    idTeamAway = int(idTeamAway) if idTeamAway is not None else None

    print("######### Treinando Partida ##########")
    fixturesRegras.fixturesModel.atualizarDados(arr_ids_team=[idTeamHome, idTeamAway])

    try:

        #previsaoFFFHome = iaLTSM.preverComFF(id_team_home=idTeamHome, id_team_away=idTeamAway, id_season=idSeason,
                                             #isPartida=True)

        previsaoRNNHome = iaLTSM.preverComRNN(id_team_home=idTeamHome, id_season=idSeason, isPartida=True, qtdeDados=50)
        previsaoRNNAWAY = iaLTSM.preverComRNN(id_team_home=idTeamAway, id_season=idSeason, isPartida=True, qtdeDados=50)
        #previsaoRNNAmbas = iaLTSM.preverComRNN(id_team_home=idTeamHome, id_team_away=idTeamAway, id_season=idSeason,
                                               #isPartida=True, qtdeDados=120, isAmbas=True)

        #print("Previsao FFHome", previsaoFFFHome)
        print("Previsao RNN home", previsaoRNNHome)
        print("Previsao RNN away", previsaoRNNAWAY)
        #print("Previsao RNN ambas", previsaoRNNAmbas)
    except Exception as exc:
        print(exc)
        return JsonResponse({"erro": "Não consegui obter a relação entre esses dois times,"
                                     " não se preocupe até o dia do jogo terei as informações."}, safe=False)

    dictPrevPartida = {
        "v_ia": "0.35.1",
        "erro": previsao.msg_erro,
        "qtde_dados_home": previsao.qtde_dados_entrada,
        "qtde_dados_away": previsao.qtde_dados_entrada,
        "data_jogo_previsto": previsao.data_previsao
    }

    dictPrevPartida["previsao_home"] = {
        "vitoria": f"{previsao.previsao[0][0][0] * 100:.2f}%",
        "empate": f"{previsao.previsao[0][0][1] * 100:.2f}%",
        "derrota": f"{previsao.previsao[0][0][2] * 100:.2f}%"
    }

    dictPrevPartida["previsao_away"] = {
        "vitoria": f"{previsao.previsao[0][1][2] * 100:.2f}%",
        "empate": f"{previsao.previsao[0][1][1] * 100:.2f}%",
        "derrota": f"{previsao.previsao[0][1][0] * 100:.2f}%"
    }


    print("######### Previsões ##########")
    name_team_home = statisticsRegras.teamsRegras.teamsModel.obterByColumnsID(arrDados=[idTeamHome])[0].name
    name_team_away = statisticsRegras.teamsRegras.teamsModel.obterByColumnsID(arrDados=[idTeamAway])[0].name
    #print("Team home: ", name_team_home, "Team away: ", name_team_away)
    print(dictPrevPartida["previsao_home"], dictPrevPartida["previsao_away"])
    database.closeConnection()
    return JsonResponse({"response": dictPrevPartida}, safe=False)

def obterPrevisaoTeam(request):
    pass
    iaRegras = IAUteisRegras()
    iaLTSM = RedeLTSM()
    uteisRegras = UteisRegras()
    statisticsRegras = StatisticsRegras()
    fixturesRegras = FixturesRegras()
    idSeason = request.GET.get("id_season")
    idTeam = request.GET.get("id_team")

    if idSeason is None and idTeam is None:
        raise "É necessário passar o prametro id_season ou id_team"

    print("############## new Request #######################")
    idSeason = int(idSeason)
    idTeam = int(idTeam)

    fixturesRegras.fixturesModel.atualizarDados(arr_ids_team=[idTeam])

    print("######### Treinando Team ##########")
    try:
        previsao: ModelPrevisao = iaLTSM.preverComRNN(id_team_home=idTeam, id_season=idSeason, qtdeDados=35)
    except Exception as exc:
        print(exc)
        return JsonResponse({"erro": "Não consegui obter a relação entre esses dois times,"
                                     " não se preocupe até o dia do jogo terei as informações."}, safe=False)

    dictPrevPartida = {
        "v_ia": "0.35.1",
        "erro": previsao.msg_erro,
        "qtde_dados": previsao.qtde_dados_entrada,
        "data_jogo_previsto": previsao.data_previsao
    }

    dictPrevPartida["previsao"] = {
        "vitoria": f"{previsao.previsao[0][0][2] * 100:.2f}%" if previsao.previsao[0][0][2] > 0  else 0,
        "empate": f"{previsao.previsao[0][0][1] * 100:.2f}%" if previsao.previsao[0][0][1] > 0  else 0,
        "derrota": f"{previsao.previsao[0][0][0] * 100:.2f}%" if previsao.previsao[0][0][0] > 0  else 0
    }

    print("######### Previsões ##########")
    name_team_home = statisticsRegras.teamsRegras.teamsModel.obterByColumnsID(arrDados=[idTeam])[0].name
    #print("Team home: ", name_team_home)
    print(dictPrevPartida["previsao"])
    database.closeConnection()
    return JsonResponse({"response": dictPrevPartida}, safe=False)

def obterTabelaPontuacao(request):
    fixturesRegras = FixturesRegras()
    teamRegras = TeamsRegras()
    uteisRegras = UteisRegras()
    tabelaPontucaoRegras = TabelaPontuacaoRegras()

    idSeason = request.GET.get("id_season")

    if idSeason is None:
        raise "Sem id_season como parametro para obter tabela pontuacao"

    teamRegras.teamsModel.atualizarDados(id_season=idSeason)
    fixturesRegras.fixturesModel.atualizarDados(id_season=idSeason)
    tabelaPontuacao = tabelaPontucaoRegras.obterTabelaPontucao(id_season=idSeason)
    tabelaPontuacaoNormalizada = uteisRegras.normalizarDadosForView(arrDados=[tabelaPontuacao])[0]
    return JsonResponse({"response": tabelaPontuacaoNormalizada}, safe=True)

def obterTabelaJogos(request):
    fixturesRegras = FixturesRegras()
    tabelaJogosRegras = TabelaJogosRegras()
    teamRegras = TeamsRegras()
    uteisRegras = UteisRegras()

    idSeason = request.GET.get("id_season")

    if idSeason is None:
        raise "Sem id_season como parametro para obter tabela pontuacao"

    teamRegras.teamsModel.atualizarDados(id_season=idSeason)
    fixturesRegras.fixturesModel.atualizarDados(id_season=idSeason)
    tabelajogos = tabelaJogosRegras.obterTabelaJogos(id_season=idSeason)
    tabelaJogosNormalizada = uteisRegras.normalizarDadosForView(arrDados=[tabelajogos])[0]
    return JsonResponse({"response": tabelaJogosNormalizada}, safe=True)

def obterEstatisticas(request):
    datasetFixtureRegras = DatasetFixtureRegras()
    uteisRegras = UteisRegras()
    id_team = request.GET.get("id_team")

    if id_team is None:
        raise "Sem id_season como parametro para obter tabela pontuacao"

    arrRetorno = datasetFixtureRegras.criarDatasetFixture(arr_ids_team=[int(id_team)], isFiltrarApenasTeams=True)
    arrRetornoNormalizado = uteisRegras.normalizarDadosForView(arrDados=arrRetorno)
    return JsonResponse({"response": arrRetornoNormalizado}, safe=False)


def urlTesteMetodos(request):
    iaAprendizadoLTSM = RedeLTSM()

    previstos = RedeLTSM.prever()

    return JsonResponse({"sfdsf": "sdfds"}, safe=False)