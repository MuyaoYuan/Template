import argparse
import template

parser = argparse.ArgumentParser(description='transfer learning template')

parser.add_argument('-tp','--template', default='.',
                    help='You can set various templates in template.py')
parser.add_argument('-t','--task', default='.',
                    help='preparation, train, reload-pre or reload-trained')

args = parser.parse_args()
template.set_template(args)