# This file contains variabels that are shared accross preprocessing scripts

test_variable = "Hello World"
country_list = ["de", "at", "ch"]
city_list = ["Berlin", "Hamburg", "Munich", "Vienna", "Zurich"]

# List of tuples indicationg important days for marketing purposes. Each tuples has the form (Start_day, End_day, Name_of_Event)
important_sales_events = [
    ('29.11.2019', '29.11.2019' , 'black_friday'), ('27.11.2020', '27.11.2020' , 'black_friday'),  ('26.11.2021', '26.11.2021' , 'black_friday'), ('25.11.2022', '25.11.2022' , 'black_friday'), ('24.11.2023', '24.11.2023' , 'black_friday'),
    ('02.12.2019', '02.12.2019' , 'cyber_monday'), ('30.11.2020', '30.11.2020' , 'cyber_monday'),  ('29.11.2021', '29.11.2021' , 'cyber_monday'), ('28.11.2022', '28.11.2022' , 'cyber_monday'), ('27.11.2023', '27.11.2023' , 'cyber_monday'),
    ('12.05.2019', '12.05.2019' , 'mothers_day'), ('10.05.2020', '10.05.2020' , 'mothers_day'),  ('09.05.2021', '09.05.2021' , 'mothers_day'), ('08.05.2022', '08.05.2022' , 'mothers_day'), ('14.05.2023', '14.05.2023' , 'mothers_day'),
    ('14.02.2019', '14.02.2019' , 'valentines_day'), ('14.02.2020', '14.02.2020' , 'valentines_day'),  ('14.02.2021', '14.02.2021' , 'valentines_day'), ('14.02.2022', '14.02.2022' , 'valentines_day'), ('14.02.2023', '14.02.2023' , 'valentines_day'),
    ('24.12.2019', '24.12.2019' , 'christmas_eve'), ('24.12.2020', '24.12.2020', 'christmas_eve'), ('24.12.2021', '24.12.2021', 'christmas_eve'), ('24.12.2022', '24.12.2022', 'christmas_eve'), ('24.12.2023', '24.12.2023', 'christmas_eve')
]

important_sales_events_list = ['black_friday',
                               'cyber_monday',
                               'mothers_day',
                               'valentines_day',
                               'christmas_eve']

secondary_sales_events = [('30.05.2019', '30.05.2019' , 'fathers_day'), ('21.05.2020', '21.05.2020' , 'fathers_day'),  ('13.05.2021', '13.05.2021' , 'fathers_day'), ('26.05.2022', '26.05.2022' , 'fathers_day'), ('18.05.2023', '18.05.2023' , 'fathers_day'),
    ('14.01.2019', '14.01.2019', 'orthodox_new_year'), ('14.01.2020', '14.01.2020', 'orthodox_new_year'), ('14.01.2021', '14.01.2021', 'orthodox_new_year'), ('14.01.2022', '14.01.2022', 'orthodox_new_year'), ('14.01.2023', '14.01.2023', 'orthodox_new_year'),
    ('05.02.2019', '05.02.2019', 'chinese_new_year'), ('25.01.2020', '25.01.2020', 'chinese_new_year'), ('12.02.2021', '12.02.2021', 'chinese_new_year'), ('01.02.2022', '01.02.2022', 'chinese_new_year'), ('22.01.2023', '22.01.2023', 'chinese_new_year'),
    ('04.03.2019', '04.03.2019', 'rosenmontag'), ('24.02.2020', '24.02.2020', 'rosenmontag'), ('15.02.2021', '15.02.2021', 'rosenmontag'), ('28.02.2022', '28.02.2022', 'rosenmontag'), ('20.02.2023', '20.02.2023', 'rosenmontag'),
    ('28.02.2019', '06.03.2019', 'carneval'), ('20.02.2020', '26.02.2020', 'carneval'), ('11.02.2021', '17.02.2021', 'carneval'), ('24.02.2022', '02.03.2022', 'carneval'), ('16.02.2023', '22.02.2023', 'carneval'),
    ('05.05.2019', '05.05.2019', 'start_of_ramadan'), ('23.04.2020', '23.04.2020', 'start_of_ramadan'), ('12.04.2021', '12.04.2021', 'start_of_ramadan'), ('02.04.2022', '02.04.2022', 'start_of_ramadan'), ('23.03.2023', '23.04.2023', 'start_of_ramadan'),
    ('14.05.2019', '14.05.2019', 'start_of_eurovision'), ('18.05.2021', '18.05.2021', 'start_of_eurovision'), ('10.05.2022', '10.05.2022', 'start_of_eurovision'), ('09.05.2023', '09.05.2023', 'start_of_eurovision'),
    ('31.10.2019', '31.10.2019', 'halloween'), ('31.10.2020', '31.10.2020', 'halloween'), ('31.10.2021', '31.10.2021', 'halloween'), ('31.10.2022', '31.10.2022', 'halloween'), ('31.10.2023', '31.10.2023', 'halloween'),
    ('05.12.2019', '05.12.2019', 'saint_nicholas'), ('05.12.2020', '05.12.2020', 'saint_nicholas'), ('05.12.2021', '05.12.2021', 'saint_nicholas'), ('05.12.2022', '05.12.2022', 'saint_nicholas') , ('05.12.2023', '05.12.2023', 'saint_nicholas')
]

secondary_sales_events_list = ['fathers_day',
                               'orthodox_new_year',
                               'chinese_new_year',
                               'rosenmontag',
                               'carneval',
                               'start_of_ramadan',
                               'start_of_eurovision',
                               'halloween',
                               'saint_nicholas']

all_special_events_list = ['black_friday',
                       'cyber_monday',
                       'mothers_day',
                       'valentines_day',
                       'christmas_eve',
                       'fathers_day',
                       'orthodox_new_year',
                       'chinese_new_year',
                       'rosenmontag',
                       'carneval',
                       'start_of_ramadan',
                       'start_of_eurovision',
                       'halloween',
                       'saint_nicholas']