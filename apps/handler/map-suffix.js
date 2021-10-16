const c_media = window.matchMedia('(max-width: 1023px)');
var names = []; var scale = []; var caps = []; var price = []; var perf = []; var color = []; var bar_data = [];
var map_type; var market_type; var option_type; var map_key;

const map_layout = {
    height: 690,
    margin:{l:2,r:2,t:2,b:25}
};
const map_option = {
    displayModeBar:false,
    responsive:true,
    showTips:false
};
const bar_option = {
		displayModeBar:false,
		responsive:true,
		showTips:false,
		staticPlot:true
};
var bar_layout = {
		height: 690,
		margin: {l: 125, r: 5, t: 10, b: 22},
		xaxis: {
			range:[0, 0],
			linecolor: 'black',
			linewidth: 1,
		},
		yaxis: {
			linecolor: 'black',
			linewidth: 1,
			tickson: 'boundaries',
			ticklen: 8,
		},
		autorange: false
	}

const arrAbs = (array) => {
    return array.map(Math.abs);
}

function setSelect(){
  $('.map-select').append('<option value="ind">산업분류</option>');
  $('.map-select').append('<option value="sec">업종분류</option>');
  $('.map-select').append('<option value="thm">주요테마</option>');
  $('.map-select').append('<option value="etf"> ETF </option>');

  $('.market-select').append('<option value="ful">전체</option>');
  $('.market-select').append('<option value="ksl">코스피 대형주</option>');
  $('.market-select').append('<option value="ksm">코스피 중형주</option>');
  $('.market-select').append('<option value="kss">코스피 소형주</option>');
  $('.market-select').append('<option value="ks2">코스피 200</option>');
  $('.market-select').append('<option value="kql">코스닥 대형주</option>');
  $('.market-select').append('<option value="kqm">코스닥 중형주</option>');
  $('.market-select').append('<option value="kq1">코스닥 150</option>');

  $('.option-select').append('<option value="R1D">1일 수익률</option>');
  $('.option-select').append('<option value="R1W">1주 수익률</option>');
  $('.option-select').append('<option value="R1M">1개월 수익률</option>');
  $('.option-select').append('<option value="R3M">3개월 수익률</option>');
  $('.option-select').append('<option value="R6M">6개월 수익률</option>');
  $('.option-select').append('<option value="R1Y">1년 수익률</option>');
  $('.option-select').append('<option value="PER">PER</option>');
  $('.option-select').append('<option value="PBR">PBR</option>');
  $('.option-select').append('<option value="DIV">배당수익률</option>');
}

function setSearch(key){
	$("#map-search").empty();
	$('#map-search').append("<option></option>");
	for (var n = 0; n < ids[key].length; n++){
		$('#map-search').append("<option>" + ids[key][n] + "</option>");
	}
}

function search_top(__group__){
  var elems = $('.slicetext');
	var clicker = null;
	var name = '';
	for (var n = 0; n < elems.length; n++){
		name = $(elems[n]).text();
		if (name == __group__){
			clicker = $(elems[n]).parent().get(0);
			break;
		}
	}
	if (clicker == null) {return;}
	if (document.createEvent) {
			var event = document.createEvent("MouseEvents");
			event.initMouseEvent("click", true, true, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
			!clicker.dispatchEvent(event);
	} else if (link.fireEvent) {
		!clicker.fireEvent("onclick");
	}
}

function search_asset(__asset_name){
	var elems = $('.slicetext');
	var clicker = null;
	for (var n = 0; n < elems.length; n++){
		var name = $(elems[n]).text();
    if ((name.includes(__asset_name)) && (name.slice(0, __asset_name.length) == __asset_name)){
      var leftover = name.slice(__asset_name.length, name.length)
      var tail_cnt = 0;
      var digits = leftover.match(/\d/g);
      if (digits != null){
  			tail_cnt += digits.length;
  		}
      if (leftover.includes("-")){ tail_cnt += 1 }
      if (leftover.includes(".")){ tail_cnt += 1 }
      if (leftover.includes("%")){ tail_cnt += 1 }
      if ((leftover.length - tail_cnt) == 0){
        clicker = $(elems[n]).parent().get(0);
  		  break;
      }
    }
	}
	if (clicker == null){
		return
	}
	if (document.createEvent) {
			var event = document.createEvent("MouseEvents");
			event.initMouseEvent("click", true, true, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
			!clicker.dispatchEvent(event);
	} else if (link.fireEvent) {
		!clicker.fireEvent("onclick");
	}
}

function reframe(code){
  names.push(frm[code]["종목명"])
  scale.push(frm[code]["크기"])
  caps.push(frm[code]["시가총액"])
  price.push(frm[code]["종가"])
  perf.push(frm[code][option_type])
  color.push(frm[code]['C' + option_type])
}

function barplot(key){
	bar_data = []
	bar[key].forEach(function(code){
		bar_data.push({
            '종목명':frm[code]['종목명'],
            '옵션': frm[code][option_type],
            '색상': frm[code]['C' + option_type]
		})
	})
	bar_data = bar_data.sort(function(a,b) {
        return a['옵션'] - b['옵션'];
	});

	names = []; perf = []; color = []
	bar_data.forEach(function(elem){
		if (map_type == 'etf'){
			if (elem['종목명'] == '레버리지'){
				elem['종목명'] = '레버리지(파생)'
			} else if (elem['종목명'] == ' 레버리지 '){
				elem['종목명'] = '레버리지(지수)'
			} else if (elem['종목명'] == '인버스'){
				elem['종목명'] = '인버스(파생)'
			} else if (elem['종목명'] == ' 인버스 '){
				elem['종목명'] = '인버스(지수)'
			}
		}
		names.push(elem['종목명'])
		perf.push(elem['옵션'])
		color.push(elem['색상'])
	})

    if (map_type == 'ind'){
        bar_layout.margin.l = c_media.matches ? 110 : 125
    } else if (map_type == 'sec') {
        bar_layout.margin.l = c_media.matches ? 80 : 88
    } else if (map_type == 'etf') {
        bar_layout.margin.l = c_media.matches ? 85 : 93
    } else if (map_type == 'thm') {
        bar_layout.margin.l = c_media.matches ? 105 : 112
    }

    var x_rng = Math.abs(perf[0]);
    perf.forEach(function(e){ x_rng = e > x_rng ? e : x_rng })
    if (c_media.matches){
        bar_layout.xaxis.range = [0, 1.3*x_rng]
    } else {
        bar_layout.xaxis.range = [0, 1.15*x_rng]
    }
	if (option_type == 'PER' || option_type == 'PBR') {
        var text = option_type
        var unit = ''
      } else {
        var text = '수익률'
        var unit = '%'
      }
	var bar_draw = {
		type:'bar',
        y:names,
		x:arrAbs(perf),
		orientation:'h',
		marker:{color:color},
		text:perf,
		texttemplate:"%{text}" + unit,
		textposition:"outside",
		hovertemplate:"섹터명: %{y}<br>" + text + ": %{x}" + unit + "<extra></extra>",
		opacity:0.9
	};
	Plotly.newPlot('myMap', [bar_draw], bar_layout, bar_option);
}

function treemap(key){
  names = []; scale = []; caps = []; price = []; perf = []; color = []
  labels[key].forEach(function(code){
	  names.push(frm[code]["종목명"])
	  scale.push(frm[code]["크기"])
	  caps.push(frm[code]["시가총액"])
	  price.push(frm[code]["종가"])
	  perf.push(frm[code][option_type])
	  color.push(frm[code]['C' + option_type])
	})
  if (option_type == 'PER' || option_type == 'PBR') {
    var text = option_type
    var unit = ''
  } else {
    var text = '수익률'
    var unit = '%'
  }

	var map_draw={
    type:'treemap',
    branchvalues:'total',
    labels:names,
    parents:covers[key],
		ids:ids[key],
		values:scale,
    meta:caps,
    customdata:price,
    text:perf,
		textposition:'middle center',
    textfont:{
      family:'NanumGothic, Nanum Gothic, monospace',
      color:'#ffffff'
    },
    texttemplate: '%{label}<br>%{text}' + unit + '<br>',
    hovertemplate: '%{label}<br>시총: %{meta}<br>종가: %{customdata}원<br>' + text + ': %{text}' + unit + '<extra></extra>',
		hoverlabel: {
      font: {
        family: 'NanumGothic, Nanum Gothic, monospace',
        color: '#ffffff'
      }
    },
	opacity: 0.9,
    marker: {
      colors: color,
      visible: true
    },
    pathbar: {"visible": true}
  }
	Plotly.newPlot('myMap', [map_draw], map_layout, map_option);
}

$(document).ready(function(){

  /* Initialize */
  setSelect();
  $(".map-select").prop('selectedIndex',0);
  $(".market-select").prop('selectedIndex',0);
  $(".option-select").prop('selectedIndex',0);
  $("#map-search").select2({placeholder:"종목명 검색..", allowClear: true});
  $('#mapbar').prop('checked', false);

  map_type = $('.map-select').val()
  market_type = $('.market-select').val()
  option_type = $('.option-select').val()
  map_key = map_type + market_type
  treemap(map_key);
  setSearch(map_key);

	// MAP-BAR Switch
	$( "#mapbar" ).click(function() {
		if ($('#mapbar').is(":checked")) { barplot(map_key) }
		else { treemap(map_key) }
	})

  // MAP type selection
  $(".map-select").on('change', function(){
    $(".market-select").empty();
    $(".option-select option[value='PER']").remove();
    $(".option-select option[value='PBR']").remove();
    $(".option-select option[value='DIV']").remove();
    map_type = $('.map-select').val()
    $(".market-select").prop('selectedIndex',0);
    $(".option-select").prop('selectedIndex',0);

    if (map_type != 'etf'){
      $('.option-select').append('<option value="PER">PER</option>');
      $('.option-select').append('<option value="PBR">PBR</option>');
      $('.option-select').append('<option value="DIV">배당수익률</option>');
    }

    $('.market-select').append('<option value="ful">전체</option>');
    if (map_type != 'etf' && map_type != 'thm') {
      $('.market-select').append('<option value="ksl">코스피 대형주</option>');
      $('.market-select').append('<option value="ksm">코스피 중형주</option>');
      $('.market-select').append('<option value="kss">코스피 소형주</option>');
      $('.market-select').append('<option value="ks2">코스피 200</option>');
      $('.market-select').append('<option value="kql">코스닥 대형주</option>');
      $('.market-select').append('<option value="kqm">코스닥 중형주</option>');
      $('.market-select').append('<option value="kq1">코스닥 150</option>');
    }
    map_key = map_type + 'ful'
		if ($('#mapbar').is(":checked")) { barplot(map_key) }
		else { treemap(map_key) }
    setSearch(map_key);
  })

	// MARKET Type selection: WICS/WI26 ONLY
  $(".market-select").on('change', function(){
    map_key = map_type + $('.market-select').val()
		if ($('#mapbar').is(":checked")) { barplot(map_key) }
		else { treemap(map_key) }
    setSearch(map_key);
  })

	// Option selection
  $(".option-select").on('change', function(){
    option_type = $(".option-select").val()
    if ($('#mapbar').is(":checked")) { barplot(map_key) }
    else { treemap(map_key) }
	if ((option_type == 'PER') || (option_type == 'PBR')){
	  $("#s_red").html('고평가')
	  $("#navy").html('평균')
	  $("#s_grn").html('저평가')
	} else if (option_type == 'DIV'){
	  $("#s_red").html('저배당')
	  $("#navy").html('평균')
	  $("#s_grn").html('고배당')
	} else {
      $("#s_red").html('하락')
	  $("#navy").html('보합')
	  $("#s_grn").html('상승')
	}
  })

  // MAP Search
  $('#map-search').on('select2:select', function (e) {
    var username = e.params.data.text;
    if (username == "") { return }

    var i_start = username.indexOf('[')
    if (i_start != -1){
      var group = username.slice(i_start + 1, username.length - 1);
      username = username.slice(0, i_start);
    } else if (group_data.includes(username)) {
      var group = username;
    } else{
      var idx = names.indexOf(username);
      var group = covers[map_key][idx];
    }

    search_top(group);
    if (username == group){return;}
    setTimeout(function(){
      search_asset(username);
    }, 1000)
  });

  // MAP Reset
  $('#map-reset').click(function(){
    treemap(map_key);
  })
})