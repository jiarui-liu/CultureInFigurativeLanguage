# Filter trivial idioms from English

python3 /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/training/mC4/filter_trivial_idioms_en.py --input /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/training/mC4/tmp_input.jsonl --output /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/training/mC4/tmp_output.jsonl --skip-llm

python3 /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/training/mC4/filter_trivial_idioms_en.py --input /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/en/idioms_merged_llm_formatted.jsonl --output /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl --skip-llm

# download and filter mc4

python3 download_and_filter_mc4.py --lang zh --build-index --index-dir /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/mC4/index_zh --index-cpus 8 --index-mem 80 --max-docs 10000 --use-infinigram-local --batch-size 10000 --tokenizer qwen3


python3 download_and_filter_mc4.py --lang en --build-index --index-dir /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/mC4/index_en --index-cpus 1 --index-mem 16 --max-docs 200 --use-infinigram-local --batch-size 150


## actual run



# check

python show_index_mappings.py --index-dir /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/mC4/index_en --idiom-file /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl