import React, { useEffect, useState } from 'react';
import { Brain, TrendingUp, CheckCircle, Clock } from 'lucide-react';

// Simple card components (inline to avoid import issues)
const Card = ({ children, className = '' }: any) => (
    <div className={`glass-card ${className}`}>{children}</div>
);

const CardHeader = ({ children }: any) => (
    <div className="p-6 pb-3">{children}</div>
);

const CardTitle = ({ children, className = '' }: any) => (
    <h3 className={`text-lg font-semibold ${className}`}>{children}</h3>
);

const CardContent = ({ children }: any) => (
    <div className="p-6 pt-0">{children}</div>
);

const Badge = ({ children, variant = 'default' }: any) => {
    const variants = {
        default: 'bg-blue-500 text-white',
        secondary: 'bg-slate-200 text-slate-700',
        outline: 'border border-slate-300 text-slate-700'
    };
    return (
        <span className={`px-2 py-1 rounded text-xs font-medium ${variants[variant as keyof typeof variants]}`}>
            {children}
        </span>
    );
};

const Table = ({ children }: any) => (
    <div className="w-full overflow-auto">
        <table className="w-full">{children}</table>
    </div>
);

const TableHeader = ({ children }: any) => <thead>{children}</thead>;
const TableBody = ({ children }: any) => <tbody>{children}</tbody>;
const TableRow = ({ children }: any) => <tr className="border-b border-slate-200">{children}</tr>;
const TableHead = ({ children }: any) => (
    <th className="text-left p-3 text-sm font-medium text-slate-600">{children}</th>
);
const TableCell = ({ children }: any) => (
    <td className="p-3 text-sm">{children}</td>
);

interface LearnedRule {
    id: number;
    name: string;
    description: string;
    action: string;
    confidence: number;
    support: number;
    accuracy: number;
    status: string;
    created_at: number;
    conditions: string[];
}

interface LearnedRulesResponse {
    rules: LearnedRule[];
    total: number;
    message: string;
}

export function LearnedRulesPanel() {
    const [rules, setRules] = useState<LearnedRule[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchLearnedRules();
    }, []);

    const fetchLearnedRules = async () => {
        try {
            const response = await fetch('http://localhost:8000/learned-rules');
            if (!response.ok) throw new Error('Failed to fetch learned rules');

            const data: LearnedRulesResponse = await response.json();
            setRules(data.rules);
            setLoading(false);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
            setLoading(false);
        }
    };

    const formatDate = (timestamp: number) => {
        return new Date(timestamp * 1000).toLocaleDateString();
    };

    if (loading) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Brain className="h-5 w-5" />
                        Learned Rules
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-muted-foreground">Loading learned rules...</p>
                </CardContent>
            </Card>
        );
    }

    if (error) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Brain className="h-5 w-5" />
                        Learned Rules
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-destructive">Error: {error}</p>
                </CardContent>
            </Card>
        );
    }

    if (rules.length === 0) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Brain className="h-5 w-5" />
                        Learned Rules
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-muted-foreground">
                        No learned rules yet. Rules are created automatically after ≥10 similar corrections.
                    </p>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5" />
                    Learned Rules
                    <Badge variant="secondary">{rules.length}</Badge>
                </CardTitle>
                <p className="text-sm text-muted-foreground">
                    Rules created from user corrections via adaptive learning
                </p>
            </CardHeader>
            <CardContent>
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Pattern</TableHead>
                            <TableHead>Action</TableHead>
                            <TableHead>Confidence</TableHead>
                            <TableHead>Support</TableHead>
                            <TableHead>Accuracy</TableHead>
                            <TableHead>Status</TableHead>
                            <TableHead>Created</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {rules.map((rule) => (
                            <TableRow key={rule.id}>
                                <TableCell>
                                    <div className="flex flex-col gap-1">
                                        <span className="font-medium">{rule.name}</span>
                                        <span className="text-xs text-muted-foreground">
                                            {rule.description}
                                        </span>
                                    </div>
                                </TableCell>
                                <TableCell>
                                    <Badge variant="outline">{rule.action}</Badge>
                                </TableCell>
                                <TableCell>
                                    <div className="flex items-center gap-1">
                                        <TrendingUp className="h-3 w-3 text-blue-500" />
                                        <span>{(rule.confidence * 100).toFixed(0)}%</span>
                                    </div>
                                </TableCell>
                                <TableCell>
                                    <div className="flex items-center gap-1">
                                        <CheckCircle className="h-3 w-3 text-green-500" />
                                        <span>{rule.support}</span>
                                    </div>
                                </TableCell>
                                <TableCell>
                                    <span className={rule.accuracy >= 0.8 ? 'text-green-600' : 'text-yellow-600'}>
                                        {(rule.accuracy * 100).toFixed(0)}%
                                    </span>
                                </TableCell>
                                <TableCell>
                                    <Badge
                                        variant={rule.status === 'production' ? 'default' : 'secondary'}
                                    >
                                        {rule.status}
                                    </Badge>
                                </TableCell>
                                <TableCell>
                                    <div className="flex items-center gap-1 text-sm text-muted-foreground">
                                        <Clock className="h-3 w-3" />
                                        {formatDate(rule.created_at)}
                                    </div>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </CardContent>
        </Card>
    );
}
