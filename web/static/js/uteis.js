function getCookiecsrftoken() {
    let nameCookie = "csrftoken"
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        let cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            let cookie = cookies[i].trim();
            if (cookie.substring(0, nameCookie.length + 1) === (nameCookie + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(nameCookie.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}


function isVazia(param){
    return param === null || param === undefined || param === ""
}