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
from api.regras.KerasNeurais import KerasNeurais
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
from api.models.nextGamesModel import NextGamesModel, NextGames

from api.models.model import Database

from api.regras.KerasSequencialWithDQN import ViviFoot
from api.regras.KerasSequencialNormal import KerasSequencialNormal

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
        teamHome: Team = fixturesRegras.fixturesModel.teamsModel.obterByColumnsID(arrDados=[idTeamHome])[0]
        teamAway: Team = fixturesRegras.fixturesModel.teamsModel.obterByColumnsID(arrDados=[idTeamAway])[0]

        season: Season = fixturesRegras.fixturesModel.teamsModel.seasonsModel.obterByColumnsID(
            arrDados=[idSeason])[0]
        league: League = fixturesRegras.fixturesModel.teamsModel.leaguesModel.obterByColumnsID(
            arrDados=[season.id_league])[0]
        country: Country = fixturesRegras.fixturesModel.teamsModel.countriesModel.obterByColumnsID(
            arrDados=[league.id_country])[0]

        strInfos = "Prevendo Jogo: " + teamHome.name + " VS " + teamAway.name + "\n"
        strInfos += "Pais: " + country.name + " - Liga: " + league.name + " - Season: " + str(
            season.year) + "\n"
        strInfos += "Data da analise: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        allResults = strInfos + "\n\n"

        footGolsHome = ViviFoot(id_team_home=idTeamHome, id_team_away=idTeamAway, isAmbas=False,
                                isAgruparTeams=False, idTypeReturn=1, isFiltrarTeams=True, isRecurrent=False,
                                funcAtiv='softmax', isPassadaTempoDupla=False)
        allResults += "Superior Home: \n"
        for k in range(len(footGolsHome.predit)):
            allResults += str([str(round(j * 100, 1)) + "%" for j in footGolsHome.predit[k]]) + "\n"

        # -------
        footGolsHomeB = ViviFoot(id_team_home=idTeamAway, id_team_away=idTeamAway, isAmbas=False,
                                 isAgruparTeams=False, idTypeReturn=1, isFiltrarTeams=True, isRecurrent=False,
                                 funcAtiv='softmax', isPassadaTempoDupla=False)
        allResults += "Superior Away: \n"
        for k in range(len(footGolsHomeB.predit)):
            allResults += str([str(round(j * 100, 1)) + "%" for j in footGolsHomeB.predit[k]]) + "\n"

        # -------
        footGolsHomeA = ViviFoot(id_team_home=idTeamHome, id_team_away=idTeamHome, isAmbas=False,
                                 isAgruparTeams=False, idTypeReturn=2, isFiltrarTeams=True, isRecurrent=False,
                                 funcAtiv='softmax', isPassadaTempoDupla=False)
        allResults += "Winner Home: \n"
        for k in range(len(footGolsHomeA.predit)):
            allResults += str([str(round(j * 100, 1)) + "%" for j in footGolsHomeA.predit[k]]) + "\n"
        # -------
        footGolsHomeAB = ViviFoot(id_team_home=idTeamAway, id_team_away=idTeamHome, isAmbas=False,
                                  isAgruparTeams=False, idTypeReturn=2, isFiltrarTeams=True, isRecurrent=False,
                                  funcAtiv='softmax', isPassadaTempoDupla=False)
        allResults += "Winner Away: \n"
        for k in range(len(footGolsHomeAB.predit)):
            allResults += str([str(round(j * 100, 1)) + "%" for j in footGolsHomeAB.predit[k]]) + "\n"
        # -------

        '''footGols = ViviFoot(id_team_home=idTeamHome, id_team_away=idTeamAway, isAmbas=True,
                            isAgruparTeams=True, idTypeReturn=2, isFiltrarTeams=False, isRecurrent=False,
                            funcAtiv='softmax', isPassadaTempoDupla=False)
        allResults += "Alls: \n"
        for k in range(len(footGols.predit)):
            allResults += str([str(round(j * 100, 1)) + "%" for j in footGols.predit[k]]) + "\n"'''

        '''# -------
        footGolsA = ViviFoot(id_team_home=idTeamAway, id_team_away=idTeamHome, isAmbas=True,
                             isAgruparTeams=False, idTypeReturn=4, isFiltrarTeams=True, isRecurrent=False,
                             funcAtiv='softmax', isPassadaTempoDupla=False)
        allResults += "Fora: \n"
        for k in range(len(footGolsA.predit)):
            allResults += str([str(round(j * 100, 1)) + "%" for j in footGolsA.predit[k]]) + "\n"
        # -------'''
        '''footGolsHomeAmbas = ViviFoot(id_team_home=idTeamHome, id_team_away=idTeamAway, isAmbas=True,
                                     isAgruparTeams=True, idTypeReturn=1, isFiltrarTeams=False, isRecurrent=False,
                                     funcAtiv='softmax', isPassadaTempoDupla=False)
        allResults += "Ambas: " + str([str(round(k * 100, 1)) + "%" for k in footGolsHomeAmbas.predit[0]]) + "\n"
        arrays.append(footGolsHomeAmbas.predit[0])'''

        allResults += "\n"

        allResults += "--------------------------------------------------------------------------- \n\n"
        print(allResults)
        with open("C:/Users/lucas/OneDrive/Documentos/Projetos/footxap/web/static/js/resultados-dqn.txt",
                  "a", encoding="utf-8") as results:
            results.write(allResults)
            results.close()

        print(allResults)
        return JsonResponse({"erro": "Não consegui obter a relação entre esses dois times,"
                                     " não se preocupe até o dia do jogo terei as informações."}, safe=False)
    except Exception as exc:
        print(exc)
        return JsonResponse({"erro": "Não consegui obter a relação entre esses dois times,"
                                     " não se preocupe até o dia do jogo terei as informações."}, safe=False)


def obterPrevisaoListaPartida(request):
    fixturesRegras = FixturesRegras()
    nextGamesModel = NextGamesModel()
    kerasSequencialNormal = KerasSequencialNormal()

    try:
        arrListGames = nextGamesModel.obterByColumns(arrNameColuns=["is_previu"], arrDados=[0])
        msg = "\n\n#######################################################################################\n"
        msg += "##################################### new List ########################################\n"
        msg += "#######################################################################################\n\n"
        with open("C:/Users/lucas/OneDrive/Documentos/Projetos/footxap/web/static/js/resultados-dqn.txt",
                  "a", encoding="utf-8") as results:
            results.write(msg)
            results.close()

        for indexJogo in range(len(arrListGames)):
            jogo: NextGames = arrListGames[indexJogo]

            print("------- Treinando Partida -------\n")
            fixturesRegras.fixturesModel.atualizarDados(arr_ids_team=[jogo.id_team_home, jogo.id_team_away])

            teamHome: Team = fixturesRegras.fixturesModel.teamsModel.obterByColumnsID(arrDados=[jogo.id_team_home])[0]
            teamAway: Team = fixturesRegras.fixturesModel.teamsModel.obterByColumnsID(arrDados=[jogo.id_team_away])[0]

            season: Season = fixturesRegras.fixturesModel.teamsModel.seasonsModel.obterByColumnsID(
                arrDados=[jogo.id_season])[0]
            league: League = fixturesRegras.fixturesModel.teamsModel.leaguesModel.obterByColumnsID(
                arrDados=[season.id_league])[0]
            country: Country = fixturesRegras.fixturesModel.teamsModel.countriesModel.obterByColumnsID(
                arrDados=[league.id_country])[0]

            strInfos = "Prevendo Jogo: " + teamHome.name + " VS " + teamAway.name + "\n"
            strInfos += "Pais: " + country.name + " - Liga: " + league.name + " - Season: " + str(
                season.year) + "\n"
            strInfos += "Data da analise: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"

            kerasSequencialNormal.gravarLogs(strInfos)
            # params
            qtdeTentativas = 3
            qtdeRedesPorTentativa = 3
            funcAtivacao = "softmax"

            # ------------------------------------------------------------------------------

            resultadoA = "\n#### Team " + teamHome.name + " - " + str(teamHome.id) + ": ####\n"
            kerasSequencialNormal.gravarLogs(resultadoA)
            prdict, arrAcertos = KerasSequencialNormal().getBestRedeNeuralByDataset(
                arrIdsTeam=[jogo.id_team_home, jogo.id_team_away], isAgruparTeams=False, idTypeReturn=3,
                isFiltrarTeams=True, isRecurrent=True, funcAtiv="sigmoid", isPassadaTempoDupla=False,
                nRedes=qtdeRedesPorTentativa, qtdeTentativas=qtdeTentativas, arrIdsExpecficos=[jogo.id_team_home],
                isDadosUmLadoSo=True)
            print("blablablabalb")
            print(prdict)
            print(arrAcertos)
            resultadoA = "\nWinner Home: \n"
            resultadoA += str(prdict) + " / " + str(arrAcertos) + "\n"
            kerasSequencialNormal.gravarLogs(resultadoA)

            resultadoAH = "\n#### Team " + teamHome.name + " - " + str(teamHome.id) + ": ####\n"
            kerasSequencialNormal.gravarLogs(resultadoAH)
            prdict, arrAcertos = KerasSequencialNormal().getBestRedeNeuralByDataset(
                arrIdsTeam=[jogo.id_team_home, jogo.id_team_away], isAgruparTeams=False, idTypeReturn=4,
                isFiltrarTeams=True, isRecurrent=True, funcAtiv="sigmoid", isPassadaTempoDupla=False,
                nRedes=qtdeRedesPorTentativa, qtdeTentativas=qtdeTentativas, arrIdsExpecficos=[jogo.id_team_home],
                isDadosUmLadoSo=True)
            print("blablablabalb")
            print(prdict)
            print(arrAcertos)
            resultadoAH = "\nEquilibrado Home: \n"
            resultadoAH += str(prdict) + " / " + str(arrAcertos) + "\n"
            kerasSequencialNormal.gravarLogs(resultadoAH)

            # ------------------------------------------------------------------------------

            resultadoB = "\n#### Team " + teamAway.name + " - " + str(teamAway.id) + ": ####\n"
            kerasSequencialNormal.gravarLogs(resultadoB)
            prdict, arrAcertos = KerasSequencialNormal().getBestRedeNeuralByDataset(
                arrIdsTeam=[jogo.id_team_away, jogo.id_team_home], isAgruparTeams=False, idTypeReturn=3,
                isFiltrarTeams=True, isRecurrent=True, funcAtiv="sigmoid", isPassadaTempoDupla=False,
                nRedes=qtdeRedesPorTentativa, qtdeTentativas=qtdeTentativas, arrIdsExpecficos=[jogo.id_team_away],
                isDadosUmLadoSo=True)

            print("blablablabalb")
            print(prdict)
            print(arrAcertos)
            resultadoB = "\nWinner Away: \n"
            resultadoB += str(prdict) + " / " + str(arrAcertos) + "\n"
            kerasSequencialNormal.gravarLogs(resultadoB)

            resultadoBA = "\n#### Team " + teamAway.name + " - " + str(teamAway.id) + ": ####\n"
            kerasSequencialNormal.gravarLogs(resultadoBA)
            prdict, arrAcertos = KerasSequencialNormal().getBestRedeNeuralByDataset(
                arrIdsTeam=[jogo.id_team_away, jogo.id_team_home], isAgruparTeams=False, idTypeReturn=4,
                isFiltrarTeams=True, isRecurrent=True, funcAtiv="sigmoid", isPassadaTempoDupla=False,
                nRedes=qtdeRedesPorTentativa, qtdeTentativas=qtdeTentativas, arrIdsExpecficos=[jogo.id_team_away],
                isDadosUmLadoSo=True)

            print("blablablabalb")
            print(prdict)
            print(arrAcertos)
            resultadoBA = "\nEquilibrado Away: \n"
            resultadoBA += str(prdict) + " / " + str(arrAcertos) + "\n"
            kerasSequencialNormal.gravarLogs(resultadoBA)

            resultado = resultadoA + resultadoAH + resultadoB + resultadoBA
            kerasSequencialNormal.gravarLogs(resultado)
            allResults = "\n---------------------------------------------------------------------------\n\n"
            kerasSequencialNormal.gravarLogs(allResults)

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