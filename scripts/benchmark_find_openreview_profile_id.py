r"""Evaluate the performance of `find_author_profile_ids`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from candidex.openreview import create_client, find_author_profile_ids
from candidex.sandbox.progressbar import make_progressbar
from candidex.schemas import AuthorAffiliation

if TYPE_CHECKING:
    from openreview.api import OpenReviewClient

console = Console()


@dataclass
class Sample:
    author: AuthorAffiliation
    profile_ids: list[str]

    def __post_init__(self) -> None:
        self.profile_ids = sorted([s.strip() for s in self.profile_ids])

    @property
    def has_profile(self) -> bool:
        """Return True if the author is expected to have an OpenReview
        profile."""
        return len(self.profile_ids) > 0


def _build_results_table(
    results: list[tuple[str, str, str, str, str]],
) -> tuple[Table, int, int, int, int]:
    """Build a Rich table from evaluation results.

    Args:
        results: List of (author, affiliation, result_label, expected, got)
                 tuples as returned by `_run_evaluation`.

    Returns:
        A tuple of (table, correct, missing, mismatch, unexpected) where the
        counts reflect the number of each outcome across all results.
    """
    table = Table(title="OpenReview Profile ID Evaluation", show_lines=True)
    table.add_column("Author", style="cyan")
    table.add_column("Affiliation", style="dim")
    table.add_column("Result", justify="center")
    table.add_column("Expected IDs", style="dim", no_wrap=False)
    table.add_column("Got IDs", no_wrap=False)

    correct = missing = mismatch = unexpected = 0

    for author, affiliation, result_label, expected, got in results:
        if "PASS" in result_label:
            correct += 1
        elif "MISSING" in result_label:
            missing += 1
        elif "UNEXPECTED" in result_label:
            unexpected += 1
        else:
            mismatch += 1

        table.add_row(author, affiliation, result_label, expected, got)

    return table, correct, missing, mismatch, unexpected


def _evaluate_sample(
    index: int,
    sample: Sample,
    client: OpenReviewClient,
) -> tuple[int, str, str, str, str, str]:
    """Evaluate a single sample by calling `find_author_profile_ids`.

    Two distinct evaluation modes apply depending on whether the author is
    expected to have an OpenReview profile:

    - If `sample.profile_ids` is non-empty, a PASS requires the returned
      IDs to be a non-empty subset of the expected IDs. A single match is
      sufficient.
    - If `sample.profile_ids` is empty, the author has no OpenReview profile
      and a PASS requires the function to return an empty list or None.
      Returning any IDs is a false positive and is reported as UNEXPECTED.

    Args:
        index:  The original index of the sample in the dataset, used to
                reassemble results in order after concurrent execution.
        sample: The `Sample` instance to evaluate.
        client: An authenticated `OpenReviewClient` instance shared across
                all samples to avoid creating a new connection per sample.

    Returns:
        A tuple of (index, author, affiliation, result_label, got, expected).
    """
    affiliation_str = sample.author.format_affiliations()
    expected_str = "; ".join(sample.profile_ids) if sample.has_profile else "[dim]none[/dim]"

    result = find_author_profile_ids(
        name=sample.author.author,
        affiliation=affiliation_str,
        email=sample.author.email,
        client=client,
    )

    got_str = "; ".join(result) if result else "[dim]—[/dim]"

    if not sample.has_profile:
        if not result:
            return (
                index,
                sample.author.author,
                affiliation_str,
                "[bold green]PASS[/bold green]",
                "[dim]—[/dim]",
                expected_str,
            )
        return (
            index,
            sample.author.author,
            affiliation_str,
            "[bold red]UNEXPECTED[/bold red]",
            f"[red]{got_str}[/red]",
            expected_str,
        )

    if result is None:
        return (
            index,
            sample.author.author,
            affiliation_str,
            "[bold red]MISSING[/bold red]",
            "[dim]—[/dim]",
            expected_str,
        )

    if not result:
        return (
            index,
            sample.author.author,
            affiliation_str,
            "[bold red]MISSING[/bold red]",
            "[dim]—[/dim]",
            expected_str,
        )

    expected_set = set(sample.profile_ids)
    got_set = set(result)

    if got_set <= expected_set:
        return (
            index,
            sample.author.author,
            affiliation_str,
            "[bold green]PASS[/bold green]",
            f"[green]{got_str}[/green]",
            expected_str,
        )

    return (
        index,
        sample.author.author,
        affiliation_str,
        "[bold red]MISMATCH[/bold red]",
        f"[red]{got_str}[/red]",
        expected_str,
    )


def _run_evaluation(
    samples: list[Sample],
    max_workers: int,
) -> list[tuple[str, str, str, str, str]]:
    """Run evaluation concurrently and return results in original sample
    order.

    Creates a single OpenReview client upfront and shares it across all
    concurrent evaluations to avoid creating a new connection per sample.
    Returns None if the client cannot be created.

    Args:
        samples:     List of `Sample` instances to evaluate.
        max_workers: Maximum number of concurrent threads.

    Returns:
        A list of (author, affiliation, result_label, expected, got)
        tuples in the same order as the input samples, or None if the
        client could not be created.
    """
    client = create_client()
    if client is None:
        console.print(
            "[bold red]Error:[/bold red] Could not create OpenReview client. Check your credentials."
        )
        return []

    results = [None] * len(samples)

    with make_progressbar() as progress:
        task = progress.add_task("Evaluating samples", total=len(samples))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluate_sample, i, sample, client): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                index, author, affiliation, result_label, got, expected = future.result()
                results[index] = (author, affiliation, result_label, expected, got)
                progress.advance(task)

    return results


def evaluate(samples: list[Sample], max_workers: int = 4) -> None:
    """Evaluate the accuracy of `find_author_profile_ids` on a dataset.

    For each sample, calls `find_author_profile_ids` and checks whether
    the returned IDs match expectations. Two evaluation modes apply:

    - Authors with `profile_ids` non-empty: PASS if the returned IDs are
      a non-empty subset of the expected IDs. MISSING if None or an empty
      list is returned. MISMATCH if the returned IDs contain IDs outside
      the expected set.
    - Authors with `profile_ids=[]`: PASS if None or an empty list is
      returned. UNEXPECTED if any IDs are returned.

    Samples are processed concurrently using a thread pool. Results are
    displayed in a Rich table in the original sample order, followed by
    a summary.

    Args:
        samples:     A list of `Sample` instances to evaluate.
        max_workers: Maximum number of concurrent threads. Defaults to 4.
    """
    results = _run_evaluation(samples, max_workers=max_workers)
    table, correct, missing, mismatch, unexpected = _build_results_table(results)

    console.print(table)
    console.print(
        f"\n[bold]Results:[/bold] "
        f"[green]{correct}/{len(samples)} correct[/green], "
        f"[red]{missing} missing[/red], "
        f"[red]{mismatch} mismatch[/red], "
        f"[red]{unexpected} unexpected[/red]"
    )


def make_dataset() -> list[Sample]:
    """Create a dataset of authors to evaluate
    `find_author_profile_ids`.

    Each sample contains an `AuthorAffiliation` with the author's name and
    a known affiliation, paired with the list of expected OpenReview profile
    IDs. Multiple IDs per sample account for authors who have more than one
    OpenReview profile. Multiple samples for the same author test robustness
    across different affiliation strings.

    Returns:
        A list of `Sample` instances for evaluation.
    """
    return [
        Sample(
            AuthorAffiliation(author="Thibaut Durand", affiliations=["LIP6"], email=None),
            profile_ids=["~Thibaut_Durand1", "~Thibaut_Durand2"],
        ),
        Sample(
            AuthorAffiliation(author="Thibaut Durand", affiliations=["SFU"], email=None),
            profile_ids=["~Thibaut_Durand1", "~Thibaut_Durand2"],
        ),
        Sample(
            AuthorAffiliation(
                author="Thibaut Durand", affiliations=["Simon Fraser University"], email=None
            ),
            profile_ids=["~Thibaut_Durand1", "~Thibaut_Durand2"],
        ),
        Sample(
            AuthorAffiliation(author="Thibaut Durand", affiliations=["Borealis AI"], email=None),
            profile_ids=["~Thibaut_Durand1", "~Thibaut_Durand2"],
        ),
        Sample(
            AuthorAffiliation(author="Thibaut Durand", affiliations=["RBC Borealis"], email=None),
            profile_ids=["~Thibaut_Durand1", "~Thibaut_Durand2"],
        ),
        Sample(
            AuthorAffiliation(
                author="Thibaut Durand", affiliations=["RBC Borealis AI"], email=None
            ),
            profile_ids=["~Thibaut_Durand1", "~Thibaut_Durand2"],
        ),
        Sample(
            AuthorAffiliation(author="Sepidehsadat Hosseini", affiliations=["SFU"], email=None),
            profile_ids=["~Sepidehsadat_Hosseini2"],
        ),
        Sample(
            AuthorAffiliation(
                author="Sepidehsadat Hosseini", affiliations=["Simon Fraser University"], email=None
            ),
            profile_ids=["~Sepidehsadat_Hosseini2"],
        ),
        Sample(
            AuthorAffiliation(
                author="Sepidehsadat Hosseini", affiliations=["RBC Borealis"], email=None
            ),
            profile_ids=["~Sepidehsadat_Hosseini2"],
        ),
        Sample(
            AuthorAffiliation(author="Greg Mori", affiliations=["SFU"], email=None),
            profile_ids=["~Greg_Mori1", "~Greg_Mori2"],
        ),
        Sample(
            AuthorAffiliation(
                author="Greg Mori", affiliations=["Simon Fraser University"], email=None
            ),
            profile_ids=["~Greg_Mori1", "~Greg_Mori2"],
        ),
        Sample(
            AuthorAffiliation(author="Greg Mori", affiliations=["RBC Borealis"], email=None),
            profile_ids=["~Greg_Mori1", "~Greg_Mori2"],
        ),
        Sample(
            AuthorAffiliation(author="Greg Mori", affiliations=["RBC Borealis", "SFU"], email=None),
            profile_ids=["~Greg_Mori1", "~Greg_Mori2"],
        ),
        Sample(
            AuthorAffiliation(author="Taylor Mordan", affiliations=["MobiLysis"], email=None),
            profile_ids=["~Taylor_Mordan1"],
        ),
        Sample(
            AuthorAffiliation(author="Taylor Mordan", affiliations=["VITA, EPFL"], email=None),
            profile_ids=["~Taylor_Mordan1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Taylor Mordan", affiliations=["MLIA, Sorbonne Université"], email=None
            ),
            profile_ids=["~Taylor_Mordan1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Matthieu Cord", affiliations=["Sorbonne Université"], email=None
            ),
            profile_ids=["~Matthieu_Cord1"],
        ),
        Sample(
            AuthorAffiliation(author="Matthieu Cord", affiliations=["Valeo"], email=None),
            profile_ids=["~Matthieu_Cord1"],
        ),
        Sample(
            AuthorAffiliation(author="Patrick Perez", affiliations=["Kyutai"], email=None),
            profile_ids=["~Patrick_Perez1", "~Patrick_Perez2"],
        ),
        Sample(
            AuthorAffiliation(author="Patrick Perez", affiliations=["Valeo"], email=None),
            profile_ids=["~Patrick_Perez1", "~Patrick_Perez2"],
        ),
        Sample(
            AuthorAffiliation(author="David Picard", affiliations=["LIGM, ENPC"], email=None),
            profile_ids=["~David_Picard1"],
        ),
        Sample(
            AuthorAffiliation(author="Nazanin Mehrasa", affiliations=["SFU"], email=None),
            profile_ids=["~Nazanin_Mehrasa2"],
        ),
        Sample(
            AuthorAffiliation(author="Nazanin Mehrasa", affiliations=["Borealis AI"], email=None),
            profile_ids=["~Nazanin_Mehrasa2"],
        ),
        Sample(
            AuthorAffiliation(author="Jiawei He", affiliations=["SFU"], email=None),
            profile_ids=["~Jiawei_He1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Jiawei He", affiliations=["Simon Fraser University"], email=None
            ),
            profile_ids=["~Jiawei_He1"],
        ),
        Sample(
            AuthorAffiliation(author="Jiawei He", affiliations=["Borealis AI"], email=None),
            profile_ids=["~Jiawei_He1"],
        ),
        Sample(
            AuthorAffiliation(author="Jiawei He", affiliations=["RBC Borealis"], email=None),
            profile_ids=["~Jiawei_He1"],
        ),
        Sample(
            AuthorAffiliation(author="Jeff Dean", affiliations=["Google"], email=None),
            profile_ids=["~Jeff_Dean1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Yoshua Bengio", affiliations=["Université de Montréal"], email=None
            ),
            profile_ids=["~Yoshua_Bengio1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Geoffrey Hinton", affiliations=["University of Toronto"], email=None
            ),
            profile_ids=["~Geoffrey_Hinton1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Yann LeCun", affiliations=["New York University"], email=None
            ),
            profile_ids=["~Yann_LeCun1"],
        ),
        Sample(
            AuthorAffiliation(author="Ilya Sutskever", affiliations=["OpenAI"], email=None),
            profile_ids=["~Ilya_Sutskever2"],
        ),
        Sample(
            AuthorAffiliation(author="Oriol Vinyals", affiliations=["Google DeepMind"], email=None),
            profile_ids=["~Oriol_Vinyals1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Percy Liang", affiliations=["Stanford University"], email=None
            ),
            profile_ids=["~Percy_Liang1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Chelsea Finn", affiliations=["Stanford University"], email=None
            ),
            profile_ids=["~Chelsea_Finn1"],
        ),
        Sample(
            AuthorAffiliation(author="Hugo Larochelle", affiliations=["Google Brain"], email=None),
            profile_ids=["~Hugo_Larochelle1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Ian Goodfellow", affiliations=["Google DeepMind"], email=None
            ),
            profile_ids=["~Ian_Goodfellow1"],
        ),
        Sample(
            AuthorAffiliation(author="Pieter Abbeel", affiliations=["UC Berkeley"], email=None),
            profile_ids=["~Pieter_Abbeel2"],
        ),
        Sample(
            AuthorAffiliation(author="Pieter Abbeel", affiliations=["amazon"], email=None),
            profile_ids=["~Pieter_Abbeel2"],
        ),
        Sample(
            AuthorAffiliation(author="Pieter Abbeel", affiliations=["covariant"], email=None),
            profile_ids=["~Pieter_Abbeel2"],
        ),
        # --- Authors from Asian universities with verified OpenReview profiles ---
        Sample(
            AuthorAffiliation(author="Jun Zhu", affiliations=["Tsinghua University"], email=None),
            profile_ids=["~Jun_Zhu2"],
        ),
        Sample(
            AuthorAffiliation(
                author="Shanghang Zhang", affiliations=["Peking University"], email=None
            ),
            profile_ids=["~Shanghang_Zhang1", "~Shanghang_Zhang2", "~Shanghang_Zhang4"],
        ),
        Sample(
            AuthorAffiliation(
                author="Baobao Chang", affiliations=["Peking University"], email=None
            ),
            profile_ids=["~Baobao_Chang1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Masashi Sugiyama", affiliations=["University of Tokyo"], email=None
            ),
            profile_ids=["~Masashi_Sugiyama1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Tatsuya Harada", affiliations=["University of Tokyo"], email=None
            ),
            profile_ids=["~Tatsuya_Harada1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Issei Sato", affiliations=["University of Tokyo"], email=None
            ),
            profile_ids=["~Issei_Sato1", "~Issei_Sato2"],
        ),
        Sample(
            AuthorAffiliation(author="Juho Lee", affiliations=["KAIST"], email=None),
            profile_ids=["~Juho_Lee2"],
        ),
        Sample(
            AuthorAffiliation(
                author="Ajit Rajwade",
                affiliations=["Indian Institute of Technology Bombay"],
                email=None,
            ),
            profile_ids=["~Ajit_Rajwade1"],
        ),
        Sample(
            AuthorAffiliation(author="Juho Lee", affiliations=["POSTECH"], email=None),
            profile_ids=["~Juho_Lee2"],
        ),
        # --- Authors from European universities with verified OpenReview profiles ---
        Sample(
            AuthorAffiliation(author="Andreas Krause", affiliations=["ETH Zurich"], email=None),
            profile_ids=["~Andreas_Krause1"],
        ),
        Sample(
            AuthorAffiliation(author="Melanie Zeilinger", affiliations=["ETH Zurich"], email=None),
            profile_ids=["~Melanie_Zeilinger1"],
        ),
        Sample(
            AuthorAffiliation(author="Siddhartha Mishra", affiliations=["ETH Zurich"], email=None),
            profile_ids=["~Siddhartha_Mishra1"],
        ),
        Sample(
            AuthorAffiliation(author="Volkan Cevher", affiliations=["EPFL"], email=None),
            profile_ids=["~Volkan_Cevher1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Yee Whye Teh", affiliations=["University of Oxford"], email=None
            ),
            profile_ids=["~Yee_Whye_Teh1", "~Yee_Whye_Teh2"],
        ),
        Sample(
            AuthorAffiliation(
                author="Tom Rainforth", affiliations=["University of Oxford"], email=None
            ),
            profile_ids=["~Tom_Rainforth1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Yingzhen Li", affiliations=["Imperial College London"], email=None
            ),
            profile_ids=["~Yingzhen_Li1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Isabel Valera", affiliations=["Saarland University"], email=None
            ),
            profile_ids=["~Isabel_Valera1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Philipp Hennig", affiliations=["University of Tübingen"], email=None
            ),
            profile_ids=["~Philipp_Hennig1"],
        ),
        Sample(
            AuthorAffiliation(
                author="Matthias Bethge", affiliations=["University of Tübingen"], email=None
            ),
            profile_ids=["~Matthias_Bethge1"],
        ),
        # --- Authors confirmed to have no OpenReview profile ---
        Sample(
            AuthorAffiliation(author="Aaaaaaaa Bbbbbbbbb", affiliations=["University"], email=None),
            profile_ids=[],
        ),
        Sample(
            AuthorAffiliation(
                author="Linus Torvalds", affiliations=["Linux Foundation"], email=None
            ),
            profile_ids=[],
        ),
        Sample(
            AuthorAffiliation(author="Satya Nadella", affiliations=["Microsoft"], email=None),
            profile_ids=[],
        ),
        Sample(
            AuthorAffiliation(author="Jensen Huang", affiliations=["NVIDIA"], email=None),
            profile_ids=[],
        ),
        Sample(
            AuthorAffiliation(
                author="Marc Andreessen", affiliations=["Andreessen Horowitz"], email=None
            ),
            profile_ids=[],
        ),
        Sample(
            AuthorAffiliation(author="Patrick Collison", affiliations=["Stripe"], email=None),
            profile_ids=[],
        ),
        Sample(
            AuthorAffiliation(author="Sam Altman", affiliations=["OpenAI"], email=None),
            profile_ids=[],
        ),
        Sample(
            AuthorAffiliation(author="Elon Musk", affiliations=["Tesla"], email=None),
            profile_ids=[],
        ),
        Sample(
            AuthorAffiliation(author="Reed Hastings", affiliations=["Netflix"], email=None),
            profile_ids=[],
        ),
        Sample(
            AuthorAffiliation(author="Brian Chesky", affiliations=["Airbnb"], email=None),
            profile_ids=[],
        ),
    ]


def main() -> None:
    r"""Define the main function."""
    samples = make_dataset()
    evaluate(samples)


if __name__ == "__main__":
    load_dotenv()

    main()
