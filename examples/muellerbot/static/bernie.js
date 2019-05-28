// Javascript has backticks and multi-line strings? If you're using Chrome it does
// TODO: This could all be done using DOM manipulation or some template library
var user_template = `
    <div class="Area"> 
        <div class="R">
            <img class="portrait" src="/generic_person.png"/>
        </div>    
        <div class="text L textL">$TEXT</div>    
    </div>
`
var bernie_template = `
    <div class="Area"> 
        <div class="L">
            <a href="https://berniesanders.com">
                <img class="portrait" src="/bernie.jpg"/>
                <div class="tooltip">Senator Robot Bernie Sanders</div>
            </a>
        </div>    
        <div class="bernieResponse text L textL">$TEXT</div>    
    </div>
`
var bernieIsThinking = `<center><img src="/loading.gif"></img></center>`
$(function() {
    $("#userInput").focus();
});

// I trust Bernie not to output XSS, but safety is always the first priority
function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
};

function askBernie() {
    // Display the user's message as a chat bubble
    var user_input = $('#userInput').val();
    $('#userInput').val('');
    $('.container').append(user_template.replace('$TEXT', escapeHtml(user_input)));
    $('.container').append(bernie_template.replace('$TEXT', bernieIsThinking));
    deleteOverflow();

    var lastBubble = $('.bernieResponse:last');
    // Kick off a repsonse from Bernie and display it when available
    // TODO: Handle timeouts from load
    $.ajax({
        url: "/ask_question",
        data: {
            'question': user_input
        },
        type: "GET",
        success: function(bernieWisdom) {
            lastBubble.html(escapeHtml(bernieWisdom));
        },
        error: function(xhr) {
            $('.container').append('<div>Error, Robot Bernie Sanders is overloaded. Try again later.</div>');
        }
    });
    return false;
}

function deleteOverflow() {
    while ($('.Area').length > 5) {
        $('.Area')[0].remove()
    }
}
