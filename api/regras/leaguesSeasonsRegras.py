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
                country = self.countriesRegras.obter(id=idCountry)[0]
                self.leaguesModel.atualizarDados()
                arrLeagues: list[League] = self.leaguesModel.obterByidCounty(idCountry=country.id)

            elif idLeague is not None:
                arrLeagues: list[League] = self.leaguesModel.obterByColumnsID(arrDados=[idLeague])

                if len(arrLeagues) == 1:
                    league: League = arrLeagues[0]
                    league.is_obter_dados = 1

                    self.leaguesModel.salvar(data=[league])
                    arrLeagues = [league]

        return arrLeagues


class SeasonsRegras:
    def __init__(self):
        self.seasonsModel = SeasonsModel()
        self.leaguesRegras = LeaguesRegras()

    def obter(self, id: int = None, idLeague: int = None) -> list[Season]:
        arrDados = []

        if idLeague is not None:
            league: League = self.leaguesRegras.obter(idLeague=idLeague)[0]
            arrDados: list[Season] = self.seasonsModel.obterByidLeague(idLeague=league.id)
        elif id is not None:
            arrDados: list[Season] = self.seasonsModel.obterByColumnsID(arrDados=[id])

        return arrDados