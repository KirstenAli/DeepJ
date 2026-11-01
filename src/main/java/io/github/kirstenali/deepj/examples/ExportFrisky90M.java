package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.publishing.DeepJModelBundle;
import io.github.kirstenali.deepj.publishing.ModelCard;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;

import java.io.IOException;
import java.nio.file.Path;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.path;

public final class ExportFrisky90M {

    private ExportFrisky90M() {}

    public static void main(String[] args) throws Exception {
        Path checkpoint = path("deepj.checkpoint", "checkpoints/deepj-90m/sft/model-final.dj");
        Path bundle = path("deepj.bundle", "dist/frisky-90m");
        export(checkpoint, bundle);
        System.out.println("Exported Frisky 90M to " + bundle);
    }

    static Path export(Path checkpoint, Path bundle) throws IOException {
        var artifacts = DeepJPrismArtifacts.load(checkpoint.getParent(), checkpoint);
        if (artifacts.model().promptFormat() != PromptFormat.INSTRUCTION_RESPONSE) {
            throw new IOException("Frisky requires an instruction-response checkpoint");
        }
        return DeepJModelBundle.export(bundle, artifacts.model(),
                BPEModelIO.load(checkpoint.getParent().resolve("tokenizer.bpe")), card());
    }

    private static ModelCard card() {
        return new ModelCard("Frisky 90M", summary(), "mit", "en",
                trainingData(), limitations());
    }

    private static String summary() {
        return "Frisky is a 90,128,896-parameter language model trained from scratch "
                + "with [DeepJ](https://github.com/KirstenAli/DeepJ) on an Apple GPU. "
                + "It completed base pretraining, mid-training, and response-only instruction fine-tuning.";
    }

    private static String trainingData() {
        return "Base pretraining used [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). "
                + "Mid-training and fine-tuning used mixtures derived from "
                + "[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), "
                + "[MMLU](https://huggingface.co/datasets/cais/mmlu), "
                + "[GSM8K](https://huggingface.co/datasets/openai/gsm8k), and "
                + "[ARC](https://huggingface.co/datasets/allenai/ai2_arc). "
                + "The MIT license on this repository applies to the model weights, "
                + "not to the upstream datasets; consult each dataset's terms.";
    }

    private static String limitations() {
        return "Frisky is an experimental small model. It can repeat itself, invent facts, "
                + "make reasoning errors, and generate incorrect code. "
                + "Its post-training quality has not yet been established by a rigorous benchmark. "
                + "Do not use it for consequential decisions.";
    }
}
