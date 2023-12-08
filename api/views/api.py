import os

import numpy
import random
from copy import deepcopy
from datetime import datetime, timedelta
from json import loads, JSONDecoder, JSONEncoder, dumps
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from operator import itemgetter

from api.models.seasonsModel import Season
from api.regras.DatasetRegras import DatasetRegras
from api.regras.countriesRegras import CountriesRegras
from api.regras.leaguesSeasonsRegras import LeaguesRegras, SeasonsRegras
from api.regras.teamsRegras import TeamsRegras
from api.regras.uteisRegras import UteisRegras
from api.regras.tabelaJogosRegras import TabelaJogosRegras
from api.regras.tabelaPontuacaoRegras import TabelaPontuacaoRegras
from api.regras.fixturesRegras import FixturesRegras
from api.datasets.datasetPartida import DatasetPartida
# Create your views here.

from api.models.countriesModel import Country
from api.models.leaguesModel import League
from api.models.teamsModel import Team
from api.models.nextGamesModel import NextGamesModel, NextGames

from api.models.model import Database
from api.ia.vivx import Vivx

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
        return JsonResponse({"erro": "Não consegui obter a relação entre esses dois times,"
                                     " não se preocupe até o dia do jogo terei as informações."}, safe=False)
    except Exception as exc:
        print(exc)
        return JsonResponse({"erro": "Não consegui obter a relação entre esses dois times,"
                                     " não se preocupe até o dia do jogo terei as informações."}, safe=False)


def obterPrevisaoListaPartida(request):
    fixturesRegras = FixturesRegras()
    nextGamesModel = NextGamesModel()
    vivx = Vivx()

    try:
        arrListGames = nextGamesModel.obterByColumns(arrNameColuns=["is_previu"], arrDados=[0],
                                                     clausulaOrder=" ORDER BY data_jogo ASC")
        msgPrev = "#############################################################\n"
        for indexJogo in range(len(arrListGames)):
            jogo: NextGames = arrListGames[indexJogo]
            arr_ids_team = [jogo.id_team_home, jogo.id_team_away]
            season = fixturesRegras.fixturesModel.seasonsModel.obterByColumnsID(arrDados=[jogo.id_season])[0]
            league = fixturesRegras.fixturesModel.leaguesModel.obterByColumnsID(arrDados=[season.id_league])[0]
            country = fixturesRegras.fixturesModel.leaguesModel.countriesModel.obterByColumnsID(
                arrDados=[league.id_country])[0]
            teamHome = fixturesRegras.teamsModel.obterByColumnsID(arrDados=[jogo.id_team_home])[0]
            teamAway = fixturesRegras.teamsModel.obterByColumnsID(arrDados=[jogo.id_team_away])[0]
            if indexJogo > 0:
                msgPrev = "------------------------------------------------------------\n"
            msgPrev += "País: {} -> Campeonato: {} - {}\n".format(country.name, league.name, str(season.year))
            msgPrev += "Prevendo para os times: {} x {}\n\n".format(teamHome.name, teamAway.name)
            msgPrev += vivx.treinarVivxByTeamOnBatch(arrIdTeam=arr_ids_team)
            msgPrev += "\n\n"
            vivx.gravarLogs(msgPrev)

            jogo.is_previu = 1
            nextGamesModel.salvar(data=[jogo.__dict__])

        return JsonResponse({"erro": "SUCESSS Lista de times prevista"}, safe=False)
    except Exception as exc:
        print(exc)
        return JsonResponse({"erro": "Não consegui obter a relação entre esses dois times,"
                                     " não se preocupe até o dia do jogo terei as informações."}, safe=False)


def addTeamsToList(request):
    fixturesRegras = FixturesRegras()
    nextGamesModel = NextGamesModel()

    idSeason = request.GET.get("id_season")
    idTeamHome = request.GET.get("id_team_home")
    idTeamAway = request.GET.get("id_team_away")

    if idSeason is None and idTeamHome is None:
        raise "É necessário passar o prametro id_season ou id_team"

    idSeason = int(idSeason)
    idTeamHome = int(idTeamHome)
    idTeamAway = int(idTeamAway) if idTeamAway is not None else None

    try:
        teamHome: Team = fixturesRegras.fixturesModel.teamsModel.obterByColumnsID(arrDados=[idTeamHome])[0]
        teamAway: Team = fixturesRegras.fixturesModel.teamsModel.obterByColumnsID(arrDados=[idTeamAway])[0]

        season: Season = fixturesRegras.fixturesModel.teamsModel.seasonsModel.obterByColumnsID(
            arrDados=[idSeason])[0]
        league: League = fixturesRegras.fixturesModel.teamsModel.leaguesModel.obterByColumnsID(
            arrDados=[season.id_league])[0]
        country: Country = fixturesRegras.fixturesModel.teamsModel.countriesModel.obterByColumnsID(
            arrDados=[league.id_country])[0]

        strInfos = "Adicionado Jogo: " + teamHome.name + " VS " + teamAway.name + "\n"
        strInfos += "Pais: " + country.name + " - Liga: " + league.name + " - Season: " + str(
            season.year) + "\n"
        strInfos += "Data da analise: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(strInfos)
        print("team Home: ", teamHome.name, " - ", teamHome.id, " / ", idTeamHome)
        print("team Away: ", teamAway.name, " - ", teamAway.id, " / ", idTeamAway)

        if idTeamHome != teamHome.id or idTeamAway != teamAway.id:
            raise Exception("Ids teams estao diferentes para adicionar")

        fixtureGame = fixturesRegras.obterProximoJogo(id_team=idTeamHome, id_team_away=idTeamAway)

        arrNewNextGame = nextGamesModel.obterByColumns(arrNameColuns=["id_fixture"], arrDados=[fixtureGame.id])
        if len(arrNewNextGame) == 0:
            newNextGame = NextGames()
        elif len(arrNewNextGame) >= 2:
            raise Exception("Dados duplicados no banco para add to list para a fixture: ", fixtureGame.id)
        else:
            newNextGame = arrNewNextGame[0]

        newNextGame.id_team_home = idTeamHome
        newNextGame.id_team_away = idTeamAway
        newNextGame.id_season = idSeason
        newNextGame.id_fixture = fixtureGame.id
        newNextGame.data_jogo = fixtureGame.date.strftime(nextGamesModel.formato_datetime_YYYY_MM_DD_H_M_S)
        newNextGame.is_previu = 0

        nextGamesModel.salvar([newNextGame])

        return JsonResponse({"sucess": "Time adicionado com successo a lista."}, safe=False)
    except Exception as exc:
        print(exc)
        return JsonResponse({"erro": exc}, safe=False)


def obterPrevisaoTeam(request):
    return JsonResponse({"response": "ksdnfklsdnf"}, safe=False)


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
    return JsonResponse({"response": "arrRetornoNormalizado"}, safe=False)


def urlTesteMetodos(request):
    return JsonResponse({"sfdsf": "sdfds"}, safe=False)