import sys
import os
import math

def tag_tr(k, data, table_title):
	s = '<table border="1" style="width:100%">' + table_title + '<tr>'

	s += '<td>' + str(k) + '</td>'
	for d in data:
		d_type, d_value = data[d]
		if d_type == 'img':
			s += '<td><a href="{}"><img src="{}" width="200px" height="200px" /></a></td>'.format(os.path.join('..', d_value), os.path.join('..', d_value))
		elif d_type == 'text':
			s += '<td>{}</td>'.format(d_value)
		else:
			s += '<td>None</td>'
	
	s += '</tr></table>'

	return s

def func(output_folder, num_per_page, input_main_folder, input_folder_list_comma, input_title_list_comma, first_hierachy, fout, prompt):
    
    input_folder_list_hierachy = input_folder_list_comma.split(':')
    input_title_list_hierachy = input_title_list_comma.split(':')
    
    if len(input_title_list_hierachy) != len(input_folder_list_hierachy):
        print('Error: len(input_title_list_hierachy) is not equal to len(input_folder_list_hierachy)')
        exit(1)

    input_folder_list = input_folder_list_hierachy[0].split(',')
    print('input_folder_list: ', input_folder_list)

    input_title_list = input_title_list_hierachy[0].split(',')
    print('input_title_list: ', input_title_list)

    num_hierachy = len(input_title_list_hierachy)
    if num_hierachy > 1:
            input_folder_list_others = input_folder_list_hierachy[1]
            input_title_list_others = input_title_list_hierachy[1]
            for i in range(2, num_hierachy):
                    input_folder_list_other += ':' + input_folder_list_hierachy[i]
                    input_title_list_other += ':' + input_title_list_hierachy[i]
    
    if not os.path.exists(input_main_folder):
            print('Error: input main folder %s does not exist!' % input_main_folder)
            exit(1)

    if first_hierachy:
            if os.path.exists(output_folder):
                    print('Error: output folder %s exists!' % output_folder)
                    exit(1)	
    
            os.mkdir(output_folder)
    
    if num_hierachy > 1:
            if not os.path.exists(os.path.join(input_main_folder, 'child')):
                    print('Error: input sub-folder %s does not exist!' % os.path.join(input_main_folder, 'child'))
                    exit(1)
    
    for i in range(len(input_folder_list)):
            if input_folder_list[i] != '' and not os.path.exists(os.path.join(input_main_folder, input_folder_list[i])):
                    print('Error: input sub-folder %s does not exist!' % os.path.join(input_main_folder, input_folder_list[i]))
                    exit(1)
    
    # write the file header and footer
    html_head = '<html><head><title>Simple Viewer</title>' + \
                    '<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.8.3/jquery.min.js"></script>' + \
                    '<script src="http://code.jquery.com/ui/1.9.2/jquery-ui.js"></script>' + \
                    '<style>.folder {display: none; }</style>' + \
                    '</head><body>'
    html_tail = '<script type="text/javascript">' + \
                    '$(\'.entry button\').click(function() {' + \
                    '$(this).parents(\'.entry\').find(\'.folder\').slideToggle(1000);})' + \
                    '</script></body></html>'
    
    table_title = '<tr><td><b>ID</b></td><td><b>File Name</b></td>'
    for i in range(len(input_title_list)):
            table_title += '<td><b>' + input_title_list[i] + '</b></td>'

    k = 0

    file_name_list = []
    if os.path.isdir(os.path.join(input_main_folder, 'ids')):
            filelist = [int(f.split('.')[0]) for f in os.listdir(os.path.join(input_main_folder, 'ids')) if f.endswith('.id')]
            filelist.sort()

            for fid in filelist:
                    ffile = open(os.path.join(input_main_folder, 'ids', str(fid)+'.id'))
                    f = ffile.readline()
                    if f[-1] == '\n':
                            f = f[:-1]
                    ffile.close()
                    file_name_list.append(f)
    else:
            filelist = [f.split('.')[0] for f in os.listdir(os.path.join(input_main_folder, input_folder_list[0])) if f.endswith('.jpg') or f.endswith('.png')]
            filelist.sort()
            file_name_list = [f for f in filelist]

    tot_page_num = math.ceil(len(file_name_list) * 1.0 / num_per_page)
    
    for f in file_name_list:

            if first_hierachy and k % num_per_page == 0:
                    out_filename = os.path.join(output_folder, str(k//num_per_page)+'.html')
                    fout = open(out_filename, 'w')
                    fout.write(html_head)
                    fout.write('<h3>Folder: {}, Page Id: {}</h3>'.format(input_folder_list, k//num_per_page))
                    
                    if os.path.exists(os.path.join(input_main_folder, 'log.txt')):
                            fave = open(os.path.join(input_main_folder, 'log.txt'))
                            lines = [line.rstrip() for line in fave]
                            ave_txt = ''
                            for line in lines[:num_log_to_show]:
                                ave_txt += '<p>' + line + '</p>'
                            ave_txt += '<p><a href="%s">See More</a></p>' % os.path.join('..', 'log.txt')
                            fout.write('<b>Average Statistics:</b> '+ave_txt)

                    page_id = k//num_per_page
                    prev_id = page_id - 1
                    next_id = page_id + 1
                    prev_click = ''
                    next_click = ''
                    if prev_id < 0:
                            prev_click = ' onclick="return false;"'
                    if next_id >= tot_page_num:
                            next_click = ' onclick="return false;"'
                    fout.write('<h3><a href="{}">First</a>&nbsp;&nbsp;&nbsp;<a href="{}"{}>Prev</a>&nbsp;&nbsp;&nbsp;<a href="{}"{}>Next</a>&nbsp;&nbsp;&nbsp;<a href="{}">Last</a></h3>'.format(str(0)+'.html', str(prev_id)+'.html', prev_click, str(next_id)+'.html', next_click, str(int(tot_page_num-1))+'.html'))
    
    
            k = k + 1
            print(prompt + 'Processing %d: %s' % (k, f))
    
            data = {}
    
            counter = 1
            data[0] = ('text', f)
            for input_folder in input_folder_list:
                    if os.path.exists(os.path.join(input_main_folder, input_folder, str(f)+'.jpg')):
                            data[counter] = ('img', os.path.join(input_main_folder, input_folder, str(f)+'.jpg'))
                    elif os.path.exists(os.path.join(input_main_folder, input_folder, str(f)+'.png')):
                            data[counter] = ('img', os.path.join(input_main_folder, input_folder, str(f)+'.png'))
                    elif os.path.exists(os.path.join(input_main_folder, input_folder, str(f)+'.txt')):
                            ftxt = open(os.path.join(input_main_folder, input_folder, str(f)+'.txt'))
                            txt = ''
                            for line in ftxt.readlines():
                                txt += '<p>' + line.rstrip() + '</p>'
                            txt += '<p><a href="%s">See More</a></p>' % os.path.join('..', input_folder, str(f)+'.txt')
                            data[counter] = ('text', txt)
                    else:
                            print('Warning: there is no .jpg/.png/.txt file whose name starts with %s!' % os.path.join(input_main_folder, input_folder, str(f)))
                            data[counter] = ('none', '')
    
                    counter += 1

            fout.write('<div class="entry">')
            fout.write(tag_tr(k, data, table_title))
    
            if num_hierachy > 1:
                    fout.write('<button style="margin: 1em 1em 1em 1em; font-size: 15px; border-radius:7px;">Click Here to See/Hide More Details</button>')
                    fout.write('<div class="folder" style="margin: 0 3em 0 3em; ">')
                    func(output_folder, 1000, os.path.join(input_main_folder, 'child', str(f)), input_folder_list_others, input_title_list_others, False, fout, prompt + '\t')
                    fout.write('</div>')
                    fout.write('<button style="margin: 1em 1em 1em 1em; font-size: 15px; border-radius:7px;">Click Here to See/Hide More Details</button>')
            fout.write('</div>')
            
            if first_hierachy and k % num_per_page == 0:
                    page_id = k//num_per_page - 1;
                    next_id = page_id + 1
                    prev_id = page_id - 1
                    prev_click = ''
                    next_click = ''
                    if prev_id < 0:
                            prev_click = ' onclick="return false;"'
                    if next_id >= tot_page_num:
                            next_click = ' onclick="return false;"'
                    #fout.write(('<h3>Color Mappings</h3><table><tr><td><p>Color Mapping for Single Material</p><a href="{}"><img src="{}"/></a></td>' + 
                    #	'<td><p>Color Mapping for Sets</p><a href="{}"><img src="{}"/></a></td></td></tr></table>').format(single_map, single_map, set_map, set_map))
                    fout.write('<h3><a href="{}">First</a>&nbsp;&nbsp;&nbsp;<a href="{}"{}>Prev</a>&nbsp;&nbsp;&nbsp;<a href="{}"{}>Next</a>&nbsp;&nbsp;&nbsp;<a href="{}">Last</a></h3>'.format(str(0)+'.html', str(prev_id)+'.html', prev_click, str(next_id)+'.html', next_click, str(int(tot_page_num-1))+'.html'))
                    fout.write(html_tail)
                    fout.close()
    
    if first_hierachy and k % num_per_page:
            page_id = k//num_per_page;
            next_id = page_id + 1
            prev_id = page_id - 1
            prev_click = ''
            next_click = ''
            if prev_id < 0:
                    prev_click = ' onclick="return false;"'
            if next_id >= tot_page_num:
                    next_click = ' onclick="return false;"'
            #fout.write(('<h3>Color Mappings</h3><table><tr><td><p>Color Mapping for Single Material</p><a href="{}"><img src="{}"/></a></td>' + 
            #	'<td><p>Color Mapping for Sets</p><a href="{}"><img src="{}"/></a></td></td></tr></table>').format(single_map, single_map, set_map, set_map))
            fout.write('<h3><a href="{}">First</a>&nbsp;&nbsp;&nbsp;<a href="{}"{}>Prev</a>&nbsp;&nbsp;&nbsp;<a href="{}"{}>Next</a>&nbsp;&nbsp;&nbsp;<a href="{}">Last</a></h3>'.format(str(0)+'.html', str(prev_id)+'.html', prev_click, str(next_id)+'.html', next_click, str(int(tot_page_num-1))+'.html'))
            fout.write(html_tail)
            fout.close()

# main 
USAGE = 'USAGE: \t\tgen_html_hierachy folder num_per_page output_folder_name input_folder_list input_title_list [optional: num_log_lines_to_show, default: 1]\n' + \
            'Parameters: \tinput_folder_list: separated by comma; hierachy separated in colon\n' + \
            '\t\tinput_title_list: separated by comma; hierachy separated in colon'

if len(sys.argv) != 6 and len(sys.argv) != 7:
	print(USAGE)
	exit(1)

main_folder = sys.argv[1]
output_folder_name = sys.argv[3]

output_folder = os.path.join(main_folder, output_folder_name)
num_per_page = int(sys.argv[2])
input_main_folder = main_folder
input_folder_list_comma = sys.argv[4]
input_title_list_comma = sys.argv[5]

num_log_to_show = 10
if len(sys.argv) > 6:
    num_log_to_show = int(sys.argv[6])

func(output_folder, num_per_page, input_main_folder, input_folder_list_comma, input_title_list_comma, True, None, '')
	
fmain = open(os.path.join(output_folder, 'index.html'), 'w')
fmain.write('<html><head><meta http-equiv="refresh" content="0; url=0.html" /></head></html>')
