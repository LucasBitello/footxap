document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById("select-country").addEventListener("change", async () => {
        await ajustarSelectLeague("select-league")
    })

    document.getElementById("select-league").addEventListener("change", async () => {
        await ajustarSelectSeason("select-season")
    })

    document.getElementById("select-season").addEventListener("change", async () => {
        let id_season = document.getElementById("select-season").value

        if(!isNaN(parseInt(id_season))){
            await callGETAPI("/seasons/teams?id_season="+id_season)
        }
    })

    await ajustarSelectCountry("select-country")
    await searchTeams(true)
    await searchTeams(false)
});

async function ajustarSelectCountry(id_html_select) {
    let arrCountries = await callGETAPI("/countries")
    let selectCountry = document.getElementById(id_html_select)

    selectCountry.innerHTML = `<option value="" selected>Selecione um país...</option>`

    for (let country of arrCountries){
        selectCountry.innerHTML += `<option value="${country["id"]}">${country["name"]}</option>`
    }
}


async function ajustarSelectLeague(id_html_select) {
    let selectLeagues = document.getElementById(id_html_select)
    let idSelectCountry = document.getElementById("select-country").value

    if (!isVazia(idSelectCountry) && !isNaN(parseInt(idSelectCountry))){
        let arrLeagues = await callGETAPI("/leagues?id_country="+idSelectCountry)

        selectLeagues.innerHTML = `<option selected>Selecione uma liga...</option>`

        for (let league of arrLeagues){
            selectLeagues.innerHTML += `<option value="${league["id"]}">${league["name"]}</option>`
        }
    }
}

async function ajustarSelectSeason(id_html_select) {
    let selectSeason = document.getElementById(id_html_select)
    let idSelectLeague = document.getElementById("select-league").value

    if (!isVazia(idSelectLeague) && !isNaN(parseInt(idSelectLeague))){
        let arrSeasons = await callGETAPI("/seasons?id_league="+idSelectLeague)

        selectSeason.innerHTML = `<option selected>Selecione uma temporada...</option>`

        for (let season of arrSeasons){
            let attrSelected = season.current === 1 ? "selected" : ""
            selectSeason.innerHTML += `<option value="${season["id"]}">${season["year"]}</option>`
        }
    }
}

async function searchTeams(isHome){
    let name_diff_item = isHome ? "-home" : "-away"
    let nameIdElementDivSearch = "div-results-team"+name_diff_item
    let nameIdElementInputSearch = "input-search-team"+name_diff_item

    let divSearchTeam = document.getElementById(nameIdElementDivSearch)
    let iptSearchTeam = document.getElementById(nameIdElementInputSearch)
    divSearchTeam.style.width = iptSearchTeam.getBoundingClientRect().width + "px"


    iptSearchTeam.addEventListener("keyup", async () => {
        let sltSeason = document.getElementById("select-season")
        if(isNaN(parseInt(sltSeason.value))){
            return;
        }
        let strParams = `?name=${iptSearchTeam.value}&id_season=${sltSeason.value}`
        callGETAPI("/teams/search"+strParams).then((response) => {
            divSearchTeam.innerHTML = ""
            for(let team of response){
                divSearchTeam.innerHTML +=
                `<a href="#" class="outline-none a-team${name_diff_item}" data-id-team="${team["id"]}" 
                    data-logo-team="${team["logo"]}" data-name-team="${team["name"]}">
                    ${team["name"]} <img src="${team["logo"]}" alt="img_team">
                </a>`
            }

            let elementsResultsSearchTeam = document.querySelectorAll(".a-team"+name_diff_item)
            for (let element of elementsResultsSearchTeam){
                element.addEventListener("click", async () => {
                    let id_team = element.getAttribute("data-id-team")
                    let logo_team = element.getAttribute("data-logo-team")
                    let name_team = element.getAttribute("data-name-team")

                    iptSearchTeam.value = name_team
                    iptSearchTeam.setAttribute("data-id-team-selected", id_team)

                    let imgTeamSelected = document.getElementById("img-team"+name_diff_item)
                    imgTeamSelected.setAttribute("src", logo_team)
                    imgTeamSelected.setAttribute("width", window.screen.width * 0.25)
                    let params = ""
                    if(!isHome){
                        let id_team_home_selected = document.getElementById("input-search-team-home")
                            .getAttribute("data-id-team-selected")

                        if(!isVazia(id_team_home_selected)){
                            params = "/statistics?id_season="+sltSeason.value+"&id_team_home="+id_team_home_selected+"&id_team_away="+id_team
                            await callGETAPI(params)
                        }
                    }
                })
            }
        })
    })

    iptSearchTeam.addEventListener("focusin", () => {
        divSearchTeam.classList.remove("hidden")
        let newEvent = new Event("keyup")
        iptSearchTeam.dispatchEvent(newEvent)
    })

    iptSearchTeam.addEventListener("focusout", () => {
        setTimeout(() => {
            let idSelected = iptSearchTeam.getAttribute("data-id-team-selected")
            if(isVazia(idSelected)){
                iptSearchTeam.value = ""
            }
            divSearchTeam.classList.add("hidden")
        }, 250)
    })
}