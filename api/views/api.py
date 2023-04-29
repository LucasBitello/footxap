import numpy

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


def atualizarSeasonsTeams(request):
    teamsRegras = TeamsRegras()
    idSeason = request.GET.get("id_season")

    if idSeason is None:
        raise "É necessário passar o prametro id_season"

    teamsRegras.teamsModel.atualizarTeamsByLeagueSeason(id_season=int(idSeason))
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


def obterStatistics(request):
    iaRegras = IARegras()
    rnn = RNN(1, [1], 1)
    uteisRegras = UteisRegras()
    statisticsRegras = StatisticsRegras()
    idSeason = request.GET.get("id_season")
    idTeamHome = request.GET.get("id_team_home")
    idTeamAway = request.GET.get("id_team_away")

    if idSeason is None and idTeamHome is None:
        raise "É necessário passar o prametro id_season ou id_team"

    print("############## new Request #######################")
    idSeason = int(idSeason)
    idTeamHome = int(idTeamHome)
    idTeamAway = int(idTeamAway) if idTeamAway is not None else None

    statisticsRegras.fixturesRegras.fixturesModel.atualizarFixturesByidTeam(id_team=idTeamHome);

    if idTeamAway is not None:
        statisticsRegras.fixturesRegras.fixturesModel.atualizarFixturesByidTeam(id_team=idTeamAway);

    arrTemsInfo = statisticsRegras.obterDadosTodasSeasonLeaguesByTeam(idTeamHome=idTeamHome, idTeamAway=idTeamAway,
                                                                      id_season=idSeason)

    arrTeamsPlay = statisticsRegras.obterAllFixturesByIdTeams(idTeamHome=idTeamHome, idTeamAway=idTeamAway,
                                                              id_season=idSeason)
    ultimass = []
    for teamPlay in arrTeamsPlay:
        if teamPlay.id_season == 157 and (teamPlay.id_team_home == 1006 or teamPlay.id_team_away == 1006):
            ultimass.append(teamPlay)

    statisticsTeamHome = statisticsRegras.obterStatisticsByTeam(arrTeamsInfoDataset=arrTemsInfo, idTeam=idTeamHome)
    statisticsTeamAway = None
    if idTeamAway is not None:
        statisticsTeamAway = statisticsRegras.obterStatisticsByTeam(arrTeamsInfoDataset=arrTemsInfo, idTeam=idTeamAway)

    arrIdsPrever = [int(idTeamHome)]
    if idTeamAway is not None:
        arrIdsPrever.append(int(idTeamAway))

    datasetTeamInfo = iaRegras.normalizarDadosTeamInfoDataset(arrTeamsInfo=arrTemsInfo, arrIdsTeamPrever=arrIdsPrever)
    rnn.treinarRNN(datasetRNN=datasetTeamInfo)
    arrTeamInfoNormalizada = [arrTeam.__dict__ for arrTeam in arrTemsInfo]

    return JsonResponse({"response": arrTeamInfoNormalizada, "TeamHome": statisticsTeamHome.__dict__,
                         "TeamAway": [] if statisticsTeamAway is None else statisticsTeamAway.__dict__}, safe=False)

def obterStatistcsTeamsPlay(request):
    iaRegras = IARegras()
    rnn = RNN(1, [1], 1)
    rbm = DBN(25, 25, 0.01)
    uteisRegras = UteisRegras()
    statisticsRegras = StatisticsRegras()
    idSeason = request.GET.get("id_season")
    idTeamHome = request.GET.get("id_team_home")
    idTeamAway = request.GET.get("id_team_away")

    if idSeason is None and idTeamHome is None:
        raise "É necessário passar o prametro id_season ou id_team"

    print("############## new Request #######################")
    idSeason = int(idSeason)
    idTeamHome = int(idTeamHome)
    idTeamAway = int(idTeamAway) if idTeamAway is not None else None

    statisticsRegras.fixturesRegras.fixturesModel.atualizarFixturesByidTeam(id_team=idTeamHome);

    if idTeamAway is not None:
        statisticsRegras.fixturesRegras.fixturesModel.atualizarFixturesByidTeam(id_team=idTeamAway);

    arrTeamsPlay = statisticsRegras.obterAllFixturesByIdTeams(idTeamHome=idTeamHome, idTeamAway=idTeamAway,
                                                              id_season=idSeason)

    arrIdsPrever = [int(idTeamHome)]
    if idTeamAway is not None:
        arrIdsPrever.append(int(idTeamAway))

    datasetTeamsPlay = iaRegras.normalizarDadosTeamsPlayDataset(arrTeamsPlays=arrTeamsPlay, arrIdsTeamPrever=arrIdsPrever)
    #datasetTeamsPlay = rbm.treinarDBN(dataset=datasetTeamsPlay)
    rnn.treinarRNN(datasetRNN=datasetTeamsPlay)

    return JsonResponse({"response": "OK"}, safe=False)