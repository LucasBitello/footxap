from api.models.leaguesModel import LeaguesModel, League
from api.models.seasonsModel import SeasonsModel, Season
from api.regras.countriesRegras import CountriesRegras

class LeaguesRegras:
    def __init__(self):
        self.leaguesModel = LeaguesModel()
        self.countriesRegras = CountriesRegras()

    def obter(self, idCountry: int = None, idLeague: int = None) -> list:
        arrLeagues: list = []

        if idCountry is None and idLeague is None:
            raise "Parametro idCountry ou idLeague é obrigatório"
        else:
            if idCountry is not None:
                self.leaguesModel.atualizarDados(id_country=idCountry)
                arrLeagues: list = self.leaguesModel.obterByColumns(arrNameColuns=["id_country"], arrDados=[idCountry])

            elif idLeague is not None:
                league: League = self.leaguesModel.obterByColumnsID(arrDados=[idLeague])[0]
                self.leaguesModel.atualizarDados(id_country=league.id_country, id_league=idLeague)

                arrLeagues: list = self.leaguesModel.obterByColumnsID(arrDados=[idLeague])

        return arrLeagues


class SeasonsRegras:
    def __init__(self):
        self.seasonsModel = SeasonsModel()
        self.leaguesRegras = LeaguesRegras()

    def obter(self, id: int = None, idLeague: int = None) -> list:
        arrDados = []

        if idLeague is not None:
            league: League = self.leaguesRegras.obter(idLeague=idLeague)[0]
            arrDados: list = self.seasonsModel.obterByColumns(arrNameColuns=["id_league"], arrDados=[league.id])
        elif id is not None:
            arrDados: list = self.seasonsModel.obterByColumnsID(arrDados=[id])

        return arrDados