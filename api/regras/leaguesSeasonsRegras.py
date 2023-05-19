from api.models.leaguesModel import LeaguesModel, League
from api.models.seasonsModel import SeasonsModel, Season
from api.regras.countriesRegras import CountriesRegras

class LeaguesRegras:
    def __init__(self):
        self.leaguesModel = LeaguesModel()
        self.countriesRegras = CountriesRegras()

    def obter(self, idCountry: int = None, idLeague: int = None) -> list[League]:
        arrLeagues: list[League] = []

        if idCountry is None and idLeague is None:
            raise "Parametro idCountry ou idLeague é obrigatório"
        else:
            if idCountry is not None:
                self.leaguesModel.atualizarDados(id_country=idCountry)
                arrLeagues: list[League] = self.leaguesModel.obterByColumns(arrNameColuns=["id_country"], arrDados=[idCountry])

            elif idLeague is not None:
                league: League = self.leaguesModel.obterByColumnsID(arrDados=[idLeague])[0]
                self.leaguesModel.atualizarDados(id_country=league.id_country, id_league=idLeague)

                arrLeagues: list[League] = self.leaguesModel.obterByColumnsID(arrDados=[idLeague])

        return arrLeagues


class SeasonsRegras:
    def __init__(self):
        self.seasonsModel = SeasonsModel()
        self.leaguesRegras = LeaguesRegras()

    def obter(self, id: int = None, idLeague: int = None) -> list[Season]:
        arrDados = []

        if idLeague is not None:
            league: League = self.leaguesRegras.obter(idLeague=idLeague)[0]
            arrDados: list[Season] = self.seasonsModel.obterByColumns(arrNameColuns=["id_league"], arrDados=[league.id])
        elif id is not None:
            arrDados: list[Season] = self.seasonsModel.obterByColumnsID(arrDados=[id])

        return arrDados