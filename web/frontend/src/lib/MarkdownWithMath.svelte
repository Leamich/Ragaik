<!-- src/lib/MarkdownWithMath.svelte -->
<script lang="ts">
	import markdownit from 'markdown-it';
	// @ts-expect-error definitions for 'markdown-it-texmath'
	import Texmath from 'markdown-it-texmath';
	import katex from 'katex';
	import 'katex/dist/katex.min.css';

	const md = markdownit({ breaks: true }).use(Texmath, {
		engine: katex,
		katexOptions: {
			throwOnError: false,
			errorColor: '#cc0000',
			displayMode: true
		},
		delimiters: ['dollars', 'brackets']
	});
	export let text: string;
	text = text.replace(/(?<!\n\n)\\\[/g, '\n\n\\[');
	let actualText = md.render(text);
</script>

<main>
	<!--eslint-disable-next-line svelte/no-at-html-tags-->
	{@html actualText}
</main>
