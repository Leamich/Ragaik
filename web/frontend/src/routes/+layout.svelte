<script lang="ts">
	import HeaderLink from '$lib/HeaderLink.svelte';
	import '../app.css';

	let { children } = $props();
</script>

<!-- Header -->
<header class="bg-white shadow-md">
	<nav class="container mx-auto px-6 py-4 flex justify-between items-center">
		<a class="text-3xl font-bold text-violet-400 text-shadow-violet-300" href="/"> RAGaik </a>
		<div class="flex space-x-4">
			<HeaderLink href="/chats" text="Чат" />
			<HeaderLink href="/contacts" text="Контакты" />
		</div>
	</nav>
</header>
{@render children()}
